'''
 * Copyright (c) 2023, Dachuan Shi.
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * For full license text, see LICENSE.txt file in the repo root
 * By Dachuan Shi
'''
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_nlvr import blip_nlvr
import utils
from utils import cosine_lr_schedule, print_params_and_flops, get_sample_norm, clear_mid_vars, sync_check, sync_dict
from data import create_dataset, create_sampler, create_loader

from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch.cuda.amp import autocast as autocast
import math
from copy import deepcopy
import torch.nn as nn

# pruneable components
PRUNE_TYPE_LIST = ['token-v', 'token-l', 'weight-c', 'weight-v', 'weight-l']
TYPE_TOKEN_V, TYPE_TOKEN_L, TYPE_WEIGHT_C, TYPE_WEIGHT_V, TYPE_WEIGHT_L = 0, 1, 2, 3, 4

def get_benefit(last_step, cur_step, last_benefit, cur_benefit):
    if last_step == 0:
        return cur_benefit
    mid = cur_step - last_step
    ratio = max(-0.08 * mid + 0.48, 0.0)
    print(mid, ratio)
    return last_benefit * ratio + cur_benefit * (1 - ratio)

def random_choose(last_steps, total_steps):
    T = 5
    values = np.array(last_steps)
    values = total_steps - values
    weights = np.exp(values / T)
    weights /= weights.sum()
    index = np.random.choice(len(values), p=weights)
    return index

def add_wanda_wrapper(model):
    handles = []
    for i in range(model.layers):
        handles.append(model.visual_encoder.blocks[i].attn.proj.register_forward_hook(get_sample_norm))
        handles.append(model.visual_encoder.blocks[i].mlp.fc2.register_forward_hook(get_sample_norm))
        handles.append(model.text_encoder.encoder.layer[i].attention.output.dense.register_forward_hook(get_sample_norm))
        handles.append(model.text_encoder.encoder.layer[i].crossattention.output.dense0.register_forward_hook(get_sample_norm))
        handles.append(model.text_encoder.encoder.layer[i].crossattention.output.dense1.register_forward_hook(get_sample_norm))
        handles.append(model.text_encoder.encoder.layer[i].output.dense.register_forward_hook(get_sample_norm))
    return handles


def update_alpha_parameters(model, layers, pi, m=0, mode=None, print_info=True):
    get_w_metric = lambda module: torch.abs(module.weight.data) * torch.sqrt(module.input_norm.reshape((1,-1)))
    standarlization = lambda x: (x - torch.mean(x)) / torch.std(x)

    if mode is None:
        importance_attn = torch.stack([
            torch.stack([get_w_metric(model.module.visual_encoder.blocks[i].attn.proj).mean(dim=0) for i in range(layers)]),
            torch.stack([get_w_metric(model.module.text_encoder.encoder.layer[i].attention.output.dense).mean(dim=0) for i in range(layers)]),
            torch.stack([get_w_metric(model.module.text_encoder.encoder.layer[i].crossattention.output.dense0).mean(dim=0) for i in range(layers)]),
            torch.stack([get_w_metric(model.module.text_encoder.encoder.layer[i].crossattention.output.dense1).mean(dim=0) for i in range(layers)]),
        ]) # 4, layer, dim
    elif mode == "v":
        importance_attn = torch.stack([
            torch.stack([get_w_metric(model.module.visual_encoder.blocks[i].attn.proj).mean(dim=0) for i in range(layers)]),
        ]) # 1, layer, dim
    elif mode == "l":
        importance_attn = torch.stack([
            torch.stack([get_w_metric(model.module.text_encoder.encoder.layer[i].attention.output.dense).mean(dim=0) for i in range(layers)]),
        ]) # 1, layer, dim
    elif mode == "c":
        importance_attn = torch.stack([
            torch.stack([get_w_metric(model.module.text_encoder.encoder.layer[i].crossattention.output.dense0).mean(dim=0) for i in range(layers)]),
            torch.stack([get_w_metric(model.module.text_encoder.encoder.layer[i].crossattention.output.dense1).mean(dim=0) for i in range(layers)]),
        ]) # 2, layer, dim
    else:
        importance_attn = None
        
    *other_dims, hidden_dim = importance_attn.shape
    importance_attn = importance_attn.reshape(*other_dims, -1, 64).mean(dim=-1) # average among each head
        
    if mode is None:
        importance_ffn = torch.stack([
            torch.stack([get_w_metric(model.module.visual_encoder.blocks[i].mlp.fc2).mean(dim=0) for i in range(layers)]),
            torch.stack([get_w_metric(model.module.text_encoder.encoder.layer[i].output.dense).mean(dim=0) for i in range(layers)])
        ])
    elif mode == "v":
        importance_ffn = torch.stack([
            torch.stack([get_w_metric(model.module.visual_encoder.blocks[i].mlp.fc2).mean(dim=0) for i in range(layers)]),
        ])
    elif mode == "l":
        importance_ffn = torch.stack([
            torch.stack([get_w_metric(model.module.text_encoder.encoder.layer[i].output.dense).mean(dim=0) for i in range(layers)])
        ])
    else:
        importance_ffn = None
        
    if importance_attn is not None:
        dist.all_reduce(importance_attn, op=dist.ReduceOp.SUM)
    if importance_ffn is not None:
        dist.all_reduce(importance_ffn, op=dist.ReduceOp.SUM)
        
    if mode != "c":
        importance_attn, importance_ffn = standarlization(importance_attn), standarlization(importance_ffn) # inherit from UPop
        importances = torch.cat([importance_attn.view(-1), importance_ffn.view(-1)])
    else:
        importance_attn = standarlization(importance_attn)
        importances = torch.cat([importance_attn.view(-1)])
        
    sorted_imp, indices = torch.sort(importances, descending=False)
    compression_weight = torch.ones_like(indices)
    compression_weight[indices < importance_attn.numel()] = 64 * 4
    compression_weight[indices >= importance_attn.numel()] = 2
    threshold = sorted_imp[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*pi))]
    def update(alpha, imp):
        mask = ((imp > threshold) | (imp >= torch.quantile(imp, 0.8)))
        mask = mask + (~mask)*m
        mask[alpha==0] = 0
        alpha.data.copy_(mask)
        
    for i in range(layers):
        if mode is None:
            update(model.module.visual_encoder.blocks[i].attn.alpha, importance_attn[0, i].reshape(1,-1,1,1))
            update(model.module.visual_encoder.blocks[i].mlp.alpha, importance_ffn[0, i].reshape(1,1,-1))
            update(model.module.text_encoder.encoder.layer[i].attention.self.alpha, importance_attn[1, i].reshape(1,-1,1,1))
            update(model.module.text_encoder.encoder.layer[i].intermediate.alpha, importance_ffn[1, i].reshape(1,1,-1))
            update(model.module.text_encoder.encoder.layer[i].crossattention.self0.alpha, importance_attn[2, i].reshape(1,-1,1,1))
            update(model.module.text_encoder.encoder.layer[i].crossattention.self1.alpha, importance_attn[3, i].reshape(1,-1,1,1)) 
        elif mode == "v":
            update(model.module.visual_encoder.blocks[i].attn.alpha, importance_attn[0, i].reshape(1,-1,1,1))
            update(model.module.visual_encoder.blocks[i].mlp.alpha, importance_ffn[0, i].reshape(1,1,-1))
        elif mode == "l":
            update(model.module.text_encoder.encoder.layer[i].attention.self.alpha, importance_attn[0, i].reshape(1,-1,1,1))
            update(model.module.text_encoder.encoder.layer[i].intermediate.alpha, importance_ffn[0, i].reshape(1,1,-1))
        elif mode == "c":
            update(model.module.text_encoder.encoder.layer[i].crossattention.self0.alpha, importance_attn[0, i].reshape(1,-1,1,1))
            update(model.module.text_encoder.encoder.layer[i].crossattention.self1.alpha, importance_attn[1, i].reshape(1,-1,1,1)) 
        del model.module.visual_encoder.blocks[i].attn.proj.input_norm
        del model.module.visual_encoder.blocks[i].mlp.fc2.input_norm
        del model.module.text_encoder.encoder.layer[i].attention.output.dense.input_norm
        del model.module.text_encoder.encoder.layer[i].crossattention.output.dense0.input_norm 
        del model.module.text_encoder.encoder.layer[i].crossattention.output.dense1.input_norm
        del model.module.text_encoder.encoder.layer[i].output.dense.input_norm
    torch.cuda.empty_cache()
    
    if print_info:
        model.module.print_compression_statistics()


def weight_importance_score(model, mode=None):
    def update_input_norm(module, weight):
        weight = weight.unsqueeze(2)
        new_inp = module.new_input
        new_bs = module.new_samples
        module.input_norm *= (module.nsamples/(module.nsamples+new_bs))
        inp_square = new_inp ** 2
        cls_token = inp_square[:,0,:]
        assert (inp_square.shape[1] - weight.shape[1] == 1) or (inp_square.shape[1] - weight.shape[1] == 2)
        if inp_square.shape[1] - weight.shape[1] == 1: # no merged tokens
            normal_tokens = inp_square[:,1:,:]
        else: # with one merged token
            normal_tokens = inp_square[:,1:-1,:]
        weighted_normal_tokens = normal_tokens * weight # * normal_tokens.shape[1]
        weighted_sum = cls_token + weighted_normal_tokens.sum(dim=1)
        module.nsamples += new_bs
        module.input_norm += (weighted_sum.sum(dim=0) / module.nsamples)
        del module.new_input
        del module.new_samples

    for i in range(model.module.layers):
        if mode is None or mode == 'v':
            update_input_norm(model.module.visual_encoder.blocks[i].attn.proj, model.module.visual_encoder.blocks[i].importance_score)
        if mode is None or mode == 'l':
            update_input_norm(model.module.text_encoder.encoder.layer[i].attention.output.dense, model.module.text_encoder.encoder.layer[i].importance_score)
        if i == model.module.layers - 1:
            if mode is None or mode == 'v':
                update_input_norm(model.module.visual_encoder.blocks[i].mlp.fc2, model.module.visual_encoder.blocks[i].top_scores)
            if mode is None or mode == 'c':
                update_input_norm(model.module.text_encoder.encoder.layer[i].crossattention.output.dense0, model.module.text_encoder.encoder.layer[i].top_scores)
                update_input_norm(model.module.text_encoder.encoder.layer[i].crossattention.output.dense1, model.module.text_encoder.encoder.layer[i].top_scores)
            if mode is None or mode == 'l':
                update_input_norm(model.module.text_encoder.encoder.layer[i].output.dense, model.module.text_encoder.encoder.layer[i].top_scores)
        else:
            if mode is None or mode == 'v':
                update_input_norm(model.module.visual_encoder.blocks[i].mlp.fc2, model.module.visual_encoder.blocks[i+1].importance_score)
            if mode is None or mode == 'c':
                update_input_norm(model.module.text_encoder.encoder.layer[i].crossattention.output.dense0, model.module.text_encoder.encoder.layer[i+1].importance_score)
                update_input_norm(model.module.text_encoder.encoder.layer[i].crossattention.output.dense1, model.module.text_encoder.encoder.layer[i+1].importance_score)
            if mode is None or mode == 'l':
                update_input_norm(model.module.text_encoder.encoder.layer[i].output.dense, model.module.text_encoder.encoder.layer[i+1].importance_score)

def train(model, train_loader, val_loader, optimizer, epoch, device, config, scaler=None, v_temp=0, l_temp=0, search=False, status=None):
    model.train()  

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.7f}'))
    metric_logger.add_meter('v_temperature', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('l_temperature', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_fdt', utils.SmoothedValue(window_size=50, fmt='{value:.4f}')) # inherit from MADTP
    metric_logger.add_meter('loss_ori', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch) if not search else 'Search Epoch: [{}]'.format(epoch)
    print_freq = 50

    len_train_loader = len(train_loader)
    for i,(image0, image1, text, targets, info) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device) 
        
        if scaler is not None:
            with autocast():
                loss_ori, loss_fdt = model(images, text, targets=targets, temperature={'v_temp': v_temp, 'l_temp': l_temp}, train=True)
                loss = loss_ori + 0.1 * loss_fdt
            if search:
                if status['prune_type'] == TYPE_WEIGHT_V:
                    weight_importance_score(model, mode="v")
                elif status['prune_type'] == TYPE_WEIGHT_L:
                    weight_importance_score(model, mode="l")
                elif status['prune_type'] == TYPE_WEIGHT_C:
                    weight_importance_score(model, mode="c")
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_ori, loss_fdt = model(images, text, targets=targets, temperature={'v_temp': v_temp, 'l_temp': l_temp}, train=True)
            loss = loss_ori + 0.1 * loss_fdt
            if search:
                if status['prune_type'] == TYPE_WEIGHT_V:
                    weight_importance_score(model, mode="v")
                elif status['prune_type'] == TYPE_WEIGHT_L:
                    weight_importance_score(model, mode="l")
                elif status['prune_type'] == TYPE_WEIGHT_C:
                    weight_importance_score(model, mode="c")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())  
        metric_logger.update(loss_ori=loss_ori.item())  
        metric_logger.update(loss_fdt=loss_fdt.item())
        metric_logger.update(v_temperature=v_temp)  
        metric_logger.update(l_temperature=l_temp)  
        
        step = epoch*len_train_loader+i+1    
        if search:
            if sync_check(status['cur_gflops']-status['tar_gflops']<=2.0) == 1: # reached flops budget, stop searching
                continue

            if status['prune_type'] == TYPE_TOKEN_V:
                if step % 300 == 1:
                    status['v_temp'] += 0.4
                    v_temp = status['v_temp']
            elif status['prune_type'] == TYPE_TOKEN_L:
                if step % 300 == 1:
                    status['l_temp'] += 0.4
                    l_temp = status['l_temp']
            elif status['prune_type'] == TYPE_WEIGHT_V:
                if step % 300 == 50:
                    status['cur_weight_v_ratio'] += 0.02
                if step % 50 == 0 and step % 300 != 0:
                    update_alpha_parameters(model, 12 if config['vit']=='base' else 24, status['cur_weight_v_ratio'], 1-float(step%300)/250, "v")
            elif status['prune_type'] == TYPE_WEIGHT_L:
                if step % 300 == 50:
                    status['cur_weight_l_ratio'] += 0.02
                if step % 50 == 0 and step % 300 != 0:
                    update_alpha_parameters(model, 12 if config['vit']=='base' else 24, status['cur_weight_l_ratio'], 1-float(step%300)/250, "l")
            elif status['prune_type'] == TYPE_WEIGHT_C:
                if step % 300 == 50:
                    status['cur_weight_c_ratio'] += 0.02
                if step % 50 == 0 and step % 300 != 0:
                    update_alpha_parameters(model, 12 if config['vit']=='base' else 24, status['cur_weight_c_ratio'], 1-float(step%300)/250, "c")
            else:
                raise NotImplementedError
            
            if step % 300 == 0:
                clear_mid_vars(model)
                cur_prune_type = status['prune_type']
                status['exec_times'][cur_prune_type] += 1
                last_gflops = status['cur_gflops']
                last_val_acc = status['val_acc']
                torch.cuda.empty_cache()
                val_stats, cur_gflops = evaluate(model.module, val_loader, device, config, v_temp=v_temp, l_temp=l_temp)
                torch.cuda.empty_cache()
                model.train()
                val_acc = float(val_stats['acc'])
                delta_acc = torch.tensor(last_val_acc - val_acc, device=device)
                delta_flops = torch.tensor(last_gflops - cur_gflops, device=device)
                dist.all_reduce(delta_acc, op=dist.ReduceOp.SUM)
                dist.all_reduce(delta_flops, op=dist.ReduceOp.SUM)
                benefit = (delta_acc / (delta_flops+1e-8)).item()

                if sync_check(delta_flops>0) == 1:
                    status['benefit_list'][cur_prune_type] = get_benefit(status['last_steps'][cur_prune_type], sum(status['exec_times']), status['benefit_list'][cur_prune_type], benefit)
                    status['last_steps'][status['prune_type']] = sum(status['exec_times'])
                else:
                    if status['benefit_list'][cur_prune_type] < 0:
                        status['benefit_list'][cur_prune_type] = status['benefit_list'][cur_prune_type] * 0.5
                    else:
                        status['benefit_list'][cur_prune_type] = status['benefit_list'][cur_prune_type] * 1.5
                if dist.get_rank() == 0:
                    if random.random() < 0.2:
                        is_random = True
                        status['prune_type'] = random_choose(status['last_steps'], sum(status['exec_times']))
                    else:
                        is_random = False
                        min_idx = status['benefit_list'].index(min(status['benefit_list']))
                        status['prune_type'] = min_idx
                    status['cur_gflops'] = cur_gflops
                    status['val_acc'] = val_acc
                status = sync_dict(status, src=0)
                if utils.is_main_process():
                    print("Prune Status:", status)
                    print("Current Prune Type:", PRUNE_TYPE_LIST[status['prune_type']], "is random:", is_random)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    res = {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    res['status'] = status
    return res

@torch.no_grad()
def evaluate(model, data_loader, device, config, v_temp=0, l_temp=0, use_tool=False):
    # test
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
    GFLOPS = 0
    len_data_loader = len(data_loader)

    for image0, image1, text, targets, info in metric_logger.log_every(data_loader, print_freq, header):
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)   
        
        prediction = model(images, text, targets=targets, temperature={'v_temp': v_temp, 'l_temp': l_temp}, train=False)  
 
        _, pred_class = prediction.max(1)
        accuracy = (targets==pred_class).sum() / targets.size(0)
        
        metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))
        
        if use_tool:
            flops = FlopCountAnalysis(model.to(device), inputs=(images, text, targets, {'v_temp': v_temp, 'l_temp': l_temp}, False,))
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            flops.tracer_warnings("none")
            B = targets.shape[0] 
            GFLOPS += flops.total() / B
        else:
            GFLOPS += model.get_flops()
    GFLOPS = GFLOPS / len_data_loader / 1e9
    print("Current V Temperature:", v_temp)
    print("Current L Temperature:", l_temp)
    print("Averaged GFLOPS:", GFLOPS)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, GFLOPS

def main(args, config):
    debug = False # TODO: change to False

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    args.world_size = utils.get_world_size() if args.distributed else 1

    config['pretrained'] = args.pretrained
    config['max_epoch'] = args.epoch
    config['p'] = args.p

    #### Dataset #### 
    print("Creating nvlr dataset")
    datasets = create_dataset('nlvr', config) 
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True,False,False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]
    
    batch_size=[config['batch_size_train'],config['batch_size_test'],config['batch_size_test']]
    train_loader, val_loader, test_loader = create_loader(datasets, samplers, batch_size=batch_size, num_workers=[4,4,4], is_trains=[True, False, False], collate_fns=[None,None,None])
    
    if debug:
        from torch.utils.data import DataLoader
        from torch.utils.data import Subset
        sub_train_dataset = Subset(datasets[0], list(range(10000)))
        sampler = torch.utils.data.DistributedSampler(sub_train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        train_loader = DataLoader(
            sub_train_dataset,
            batch_size=batch_size[0],
            num_workers=4,
            pin_memory=True,
            sampler=sampler,
            shuffle=(samplers[0] is None),
            collate_fn=None,
            drop_last=True,
            prefetch_factor=4, 
            persistent_workers=True,
        )

    #### Model ####
    print("Creating model from pretrained checkpoint")
    model = blip_nlvr(pretrained=config['pretrained'], image_size=config['image_size'], 
                    vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], config=config)
    
    model = model.to(device)
    
    # print_params_and_flops('nlvr', model, device) # inherited from UPop
    # flops = model.module.get_flops() / 1e9
    # print("Handcrafted FLOPs:", flops)

    if args.evaluate and not args.searched and not args.finetuned:
        print("Evaluating the full model")
        val_stats, Cur_Gflops = evaluate(model_without_ddp, val_loader, device, config, v_temp=0, l_temp=0)
        test_stats, _ = evaluate(model_without_ddp, test_loader, device, config, v_temp=0, l_temp=0)
        if utils.is_main_process():  
            log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'GFLOPs': round(Cur_Gflops, 2),
                        }
            print(log_stats)
        exit()

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    optimizer = torch.optim.AdamW(
            params=[{'params':[param for name, param in list(model.named_parameters()) if 'alpha' not in name]}], 
            lr=config['init_lr'], 
            weight_decay=config['weight_decay']
            )
    scaler = torch.cuda.amp.GradScaler() if (not args.evaluate and args.amp) else None

    Ori_Gflops = 132.54
    Target_Gflops = Ori_Gflops * (1 - config['p'])
    print("Original model Gflops:", Ori_Gflops)
    print("Target model Gflops:", Target_Gflops)
    print('Target compression ratio: {}%'.format(config['p']*100))

    if not args.searched:
        print("Start searching (pruning)")
        Cur_Gflops = Ori_Gflops
        # An warmup to work with MADTP (otherwise, the initial state would deviate from the converged state)
        if args.firstfit is None:
            v_temp, l_temp = 0.001, 0.001
            for epoch in range(2):
                if args.distributed:
                    train_loader.sampler.set_epoch(epoch)
                cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
                train_stats = train(model, train_loader, val_loader, optimizer, epoch, device, config, scaler=scaler, v_temp=v_temp, l_temp=l_temp, search=False)
                torch.cuda.empty_cache()
                val_stats, Cur_Gflops = evaluate(model_without_ddp, val_loader, device, config, v_temp=v_temp, l_temp=l_temp)
                test_stats, _ = evaluate(model_without_ddp, test_loader, device, config, v_temp=v_temp, l_temp=l_temp)
                torch.cuda.empty_cache()
                if utils.is_main_process():  
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'val_{k}': v for k, v in val_stats.items()},
                                **{f'test_{k}': v for k, v in test_stats.items()},
                                'epoch': epoch, 'Cur_Gflops': round(Cur_Gflops, 2),
                                }
                    print("LOG: ", log_stats)
                dist.barrier()
                torch.cuda.empty_cache()
            if utils.is_main_process():  
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'v_temp': v_temp,
                    'l_temp': l_temp,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'firstfit_ckpt.pth')) 
        else:
            checkpoint = torch.load(args.firstfit)
            v_temp, l_temp = checkpoint['v_temp'], checkpoint['l_temp']
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            val_stats, Cur_Gflops = evaluate(model_without_ddp, val_loader, device, config, v_temp=v_temp, l_temp=l_temp)
            test_stats, _ = evaluate(model_without_ddp, test_loader, device, config, v_temp=v_temp, l_temp=l_temp)  
            if utils.is_main_process():  
                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'Cur_Gflops': round(Cur_Gflops, 2), 'V Temperature': v_temp, 'L Temperature': l_temp,
                            }
                print("LOG: ", log_stats)
        dist.barrier()
        torch.cuda.empty_cache()

        prune_status_mem = {
            'cur_gflops': Cur_Gflops,
            'tar_gflops': Target_Gflops,
            'prune_type': TYPE_TOKEN_L,
            'benefit_list': [0.0, 0.0, 0.0, 0.0, 0.0],
            'exec_times': [0, 0, 0, 0, 0],
            'last_steps': [0, 0, 0, 0, 0], 
            'val_acc': float(val_stats['acc']),
            'cur_weight_v_ratio': 0.0,
            'cur_weight_l_ratio': 0.0,
            'cur_weight_c_ratio': 0.0,
            'v_temp': v_temp,
            'l_temp': l_temp,
        }
        
        handles = add_wanda_wrapper(model.module)

        epoch = 0
        while epoch < config['max_epoch']:
            print("V Temperature:", v_temp, "L Temperature:", l_temp)
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train(model, train_loader, val_loader, optimizer, epoch, device, config, scaler=scaler, v_temp=v_temp, l_temp=l_temp, search=True, status=prune_status_mem)
            prune_status_mem = train_stats['status']
            v_temp, l_temp = prune_status_mem['v_temp'], prune_status_mem['l_temp']
            Cur_Gflops = train_stats['status']['cur_gflops']

            dist.barrier()
            torch.cuda.empty_cache()

            epoch += 1
            stop_training = (Cur_Gflops - Target_Gflops) < 2.0
            stop_tensor = torch.tensor(int(stop_training), device="cuda")
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                print("End Searching, cur:", Cur_Gflops, "tar:", Target_Gflops)
                break
        torch.cuda.empty_cache()

        val_stats, Cur_Gflops = evaluate(model_without_ddp, val_loader, device, config, v_temp=v_temp, l_temp=l_temp)
        test_stats, _ = evaluate(model_without_ddp, test_loader, device, config, v_temp=v_temp, l_temp=l_temp)  
        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_{k}': v for k, v in val_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch, 'Cur_Gflops': round(Cur_Gflops, 2),
                        }
            print("LOG: ", log_stats)

        model.module.print_compression_statistics()

        for handle in handles:
            handle.remove()
        clear_mid_vars(model)

        if utils.is_main_process():  
            save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'v_temp': v_temp,
                        'l_temp': l_temp,
                        'epoch': epoch,
                    }
            torch.save(save_obj, os.path.join(args.output_dir, 'search_final_ckpt.pth'))
        dist.barrier()
        torch.cuda.empty_cache()
        # del optimizer, scaler
    else:
        print(f"Loading model from searched checkpoint {args.searched}")
        checkpoint = torch.load(args.searched)
        model_without_ddp.load_state_dict(checkpoint['model'])
        v_temp, l_temp = checkpoint["v_temp"], checkpoint["l_temp"]
        epoch = checkpoint['epoch']

        val_stats, Cur_Gflops = evaluate(model_without_ddp, val_loader, device, config, v_temp=v_temp, l_temp=l_temp)
        test_stats, _ = evaluate(model_without_ddp, test_loader, device, config, v_temp=v_temp, l_temp=l_temp)  
        if utils.is_main_process():  
            log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'Cur_Gflops': round(Cur_Gflops, 2), 'V Temperature': v_temp, 'L Temperature': l_temp,
                        }
            print("LOG: ", log_stats)
        dist.barrier()
        torch.cuda.empty_cache()

    print("Creating model for finetuning")

    if not args.finetuned:
        print("Start finetuning") # Keep full parameter with sparse masks to maintain optimizer state continuity
        best = 0
        best_epoch = 0
        while epoch < config['max_epoch']:
            if Cur_Gflops > Target_Gflops:
                if Cur_Gflops - Target_Gflops > 30:
                    v_temp += 1
                    l_temp += 1
                elif Cur_Gflops - Target_Gflops > 10:
                    v_temp += 0.5
                    l_temp += 0.5
                elif Cur_Gflops - Target_Gflops > 5:
                    v_temp += 0.25
                    l_temp += 0.25
                elif Cur_Gflops - Target_Gflops > 2:
                    v_temp += 0.1
                    l_temp += 0.1
                elif Cur_Gflops - Target_Gflops > 1:
                    v_temp += 0.05
                    l_temp += 0.05
            else:
                if Target_Gflops - Cur_Gflops > 30:
                    v_temp -= 1
                    l_temp -= 1
                elif Target_Gflops - Cur_Gflops > 10:
                    v_temp -= 0.5
                    l_temp -= 0.5
                elif Target_Gflops - Cur_Gflops > 5:
                    v_temp -= 0.25
                    l_temp -= 0.25
                elif Target_Gflops - Cur_Gflops > 2:
                    v_temp -= 0.1
                    l_temp -= 0.1
                elif Target_Gflops - Cur_Gflops > 1:
                    v_temp -= 0.05
                    l_temp -= 0.05
                else:
                    v_temp -= 0.01
                    l_temp -= 0.01
            print("V Temperature:", v_temp, "L Temperature:", l_temp)
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train(model, train_loader, val_loader, optimizer, epoch, device, config, scaler=scaler, v_temp=v_temp, l_temp=l_temp, search=False)
                
            val_stats, Cur_Gflops = evaluate(model_without_ddp, val_loader, device, config, v_temp=v_temp, l_temp=l_temp, use_tool=False)
            test_stats, _ = evaluate(model_without_ddp, test_loader, device, config, v_temp=v_temp, l_temp=l_temp, use_tool=False) 
            
            if utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in val_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch, 'Cur_Gflops': round(Cur_Gflops, 2), 'V Temperature': v_temp, 'L Temperature': l_temp,
                            }
                print("LOG: ", log_stats)
                
                if float(test_stats['acc']) > best and Cur_Gflops - Target_Gflops < 1.0:
                    with torch.no_grad():
                        save_model = deepcopy(model_without_ddp)
                    save_model = save_model.to(device)
                    save_model.compress(model_without_ddp)
                    params = sum([p.numel() for p in save_model.parameters()]) / 1e6

                    save_obj = {
                        'model': save_model.state_dict(),
                        'v_temp': v_temp,
                        'l_temp': l_temp,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'finetune_best_ckpt.pth')) 
                    del save_model

                    print("Saved current best model.", {'epoch': epoch, 'Cur_Gflops': round(Cur_Gflops, 2), 'Params(MB)': round(params, 4)})

                    best = float(test_stats['acc'])
                    best_epoch = epoch

            torch.cuda.empty_cache()
            dist.barrier()  
            epoch += 1 

        if utils.is_main_process():   
            print(f"LOG: best epoch for finetuning: {best_epoch}")
    else:
        with torch.no_grad():
            search_model_without_ddp = deepcopy(model_without_ddp)
        search_model_without_ddp = search_model_without_ddp.to(device)
        model_without_ddp.compress(search_model_without_ddp)
        print("Finished compressing the searched model")
        del search_model_without_ddp

        print(f"Loading model from finetuned checkpoint {args.finetuned}")
        checkpoint = torch.load(args.finetuned)
        model_without_ddp.load_state_dict(checkpoint['model'])
        v_temp, l_temp = checkpoint["v_temp"], checkpoint["l_temp"]
        val_stats, Cur_Gflops = evaluate(model_without_ddp, val_loader, device, config, v_temp=v_temp, l_temp=l_temp, use_tool=False)
        test_stats, _ = evaluate(model_without_ddp, test_loader, device, config, v_temp=v_temp, l_temp=l_temp, use_tool=False) 
        if utils.is_main_process(): 
            log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'Cur_Gflops': round(Cur_Gflops, 2), 'V Temperature': v_temp, 'L Temperature': l_temp,
                        }
            print("LOG: ", log_stats)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/nlvr.yaml')
    parser.add_argument('--output_dir', default='output/NLVR')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--pretrained', default='pretrained/blip/model_base_nlvr.pth', type=str)
    parser.add_argument('--epoch', default=15, type=int, help='number of epochs')
    parser.add_argument('--p', default=0.5, type=float, help='total compression ratio')
    parser.add_argument('--amp', action='store_true')

    parser.add_argument('--searched', default=None, type=str)
    parser.add_argument('--finetuned', default=None, type=str)
    parser.add_argument('--firstfit', default=None, type=str)
    
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)