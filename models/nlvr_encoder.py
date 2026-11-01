import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, device, dtype, nn
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.bert.configuration_bert import BertConfig
from models.utils import *


logger = logging.get_logger(__name__)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        
        self.config = config

    def forward(
        self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        self.token_num = seq_length

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    def get_flops(self):
        flops = 0.0 + self.token_num * self.LayerNorm.weight.shape[0] * 5
        return flops


class BertSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention, attnmask=True):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.in_q_dim = config.hidden_size
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
            self.in_k_dim = config.encoder_width
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
            self.in_k_dim = config.hidden_size

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = False   
        self.attention_map = None
        self.cls_attn = None
        
        self.alpha = nn.Parameter(torch.ones(1, self.num_attention_heads, 1, 1)) # prune in a head-wise manner
        self.attnmask = attnmask
            
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def save_cls_attn(self, cls_attn):
        self.cls_attn = cls_attn
    
    def get_cls_attn(self):
        return self.cls_attn
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self.q_token_num = hidden_states.shape[1]
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
            self.in_k_token_num = encoder_hidden_states.shape[1]
            self.attn_k_token_num = self.in_k_token_num
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
            self.in_k_token_num = hidden_states.shape[1]
            self.attn_k_token_num = key_layer.shape[2]
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            self.in_k_token_num = hidden_states.shape[1]
            self.attn_k_token_num = self.in_k_token_num

        query_layer = self.transpose_for_scores(mixed_query_layer)
        if not self.attnmask and hasattr(self, 'alpha'): # baseline from UPop
            key_layer = key_layer * self.alpha
            value_layer = value_layer * self.alpha
            query_layer = query_layer * self.alpha

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        if self.attnmask and hasattr(self, 'alpha'): # CoMP - CIT
            attention_probs = attention_probs * self.alpha
        
        if not is_cross_attention:
            self.save_attention_map(attention_probs)
        
        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)         

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        attn_out = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = attn_out.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size // 2,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)

        ## AS-ViT with takes into account the attention head diversity
        if not is_cross_attention:
            cls_attn = attention_probs[:, :, 0, 1:] # [B, H, N-1]
            head_importance = attn_out[..., 1:, :].norm(dim=-1)
            head_importance = head_importance / (head_importance.sum(dim=1, keepdim=True) + 1e-8)
            cls_attn = cls_attn * head_importance
            cls_attn = cls_attn.sum(dim=1) # [B, N-1]
            self.save_cls_attn(cls_attn)

        return outputs

    def get_flops(self):
        flops = 0.0 + self.q_token_num * self.in_q_dim * self.attention_head_size + \
                self.in_k_token_num * self.in_k_dim * self.attention_head_size * 2 + \
                self.q_token_num * self.attn_k_token_num * self.attention_head_size * 2
        if hasattr(self, 'alpha'):
            flops = flops * self.alpha.sum().item()
        else:
            flops *= self.num_attention_heads
        return flops
    
    def compress_heads(self, preserve_head_idxs, template_module):
        num_heads = len(preserve_head_idxs)
        in_features_q, in_features_k, in_features_v = self.query.in_features, self.key.in_features, self.value.in_features
        head_dim = self.attention_head_size
        preserve_channel_idxs = head_to_channel(preserve_head_idxs, head_dim)
        
        self.query = nn.Linear(in_features_q, num_heads * head_dim, bias=True)
        self.key = nn.Linear(in_features_k, num_heads * head_dim, bias=True)
        self.value = nn.Linear(in_features_v, num_heads * head_dim, bias=True)
        self.query.weight.data = template_module.query.weight.data[preserve_channel_idxs, :]
        self.query.bias.data = template_module.query.bias.data[preserve_channel_idxs]
        self.key.weight.data = template_module.key.weight.data[preserve_channel_idxs, :]
        self.key.bias.data = template_module.key.bias.data[preserve_channel_idxs]
        self.value.weight.data = template_module.value.weight.data[preserve_channel_idxs, :]
        self.value.bias.data = template_module.value.bias.data[preserve_channel_idxs]
        
        self.num_attention_heads = num_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size


class BertSelfOutput(nn.Module):
    def __init__(self, config, twin=False, merge=False):     
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)      

        if twin:
            self.dense0 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)       
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        if merge:
            self.act =  ACT2FN[config.hidden_act]
            self.merge_layer = nn.Linear(config.hidden_size * 2, config.hidden_size)
            self.merge = True
        else:
            self.merge = False

    def forward(self, hidden_states, input_tensor):
        if type(hidden_states) == list:
            self.token_num = hidden_states[0].shape[1]
            hidden_states0 = self.dense0(hidden_states[0])
            hidden_states1 = self.dense1(hidden_states[1])        
            if self.merge:  
                hidden_states = self.merge_layer(torch.cat([hidden_states0,hidden_states1],dim=-1))
            else:
                hidden_states = (hidden_states0+hidden_states1)/2
        else:
            self.token_num = hidden_states.shape[1]
            hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def get_flops(self, mask=None):
        flops = 0.0
        if hasattr(self, 'dense0'):
            if mask is None:
                in_0 = self.dense0.in_features
                in_1 = self.dense1.in_features
            else:
                in_0 = self.dense0.in_features / mask[0].numel() * mask[0].sum().item()
                in_1 = self.dense1.in_features / mask[1].numel() * mask[1].sum().item()
            flops += self.token_num * in_0 * self.dense0.out_features
            flops += self.token_num * in_1 * self.dense1.out_features
            if self.merge:
                flops += self.token_num * self.merge_layer.in_features * self.merge_layer.out_features
        else:
            if mask is None:
                in_f = self.dense.in_features
            else:
                in_f = self.dense.in_features / mask[0].numel() * mask[0].sum().item()
            flops += self.token_num * in_f * self.dense.out_features
        flops += self.token_num * self.LayerNorm.weight.shape[0] * 5
        return flops
    
    def compress_inter_channels(self, preserve_idxs, template_module, which='dense'):
        hidden_features = len(preserve_idxs)
        if which == 'dense':
            self.dense = nn.Linear(hidden_features, self.dense.out_features, bias=True)
            self.dense.weight.data = template_module.dense.weight.data[:, preserve_idxs]
            self.dense.bias.data = template_module.dense.bias.data
        elif which == 'dense0':
            self.dense0 = nn.Linear(hidden_features, self.dense0.out_features, bias=True)
            self.dense0.weight.data = template_module.dense0.weight.data[:, preserve_idxs]
            self.dense0.bias.data = template_module.dense0.bias.data
        elif which == 'dense1':
            self.dense1 = nn.Linear(hidden_features, self.dense1.out_features, bias=True)
            self.dense1.weight.data = template_module.dense1.weight.data[:, preserve_idxs]
            self.dense1.bias.data = template_module.dense1.bias.data
        else:
            raise NotImplementedError('Unknown which: {}'.format(which))
        

class BertAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_num=-1, attnmask=True):
        super().__init__()
        if is_cross_attention:
            self.self0 = BertSelfAttention(config, is_cross_attention, attnmask=attnmask)
            self.self1 = BertSelfAttention(config, is_cross_attention, attnmask=attnmask)
        else:    
            self.self = BertSelfAttention(config, is_cross_attention, attnmask=attnmask)
        self.output = BertSelfOutput(config, twin=is_cross_attention, merge=(is_cross_attention and layer_num>=6))
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        space_dict=None,
    ):   
        if type(encoder_hidden_states)==list:   
            self_outputs0 = self.self0(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states[0],
                encoder_attention_mask[0],
                past_key_value,
                output_attentions,
            )
            self_outputs1 = self.self1(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states[1],
                encoder_attention_mask[1],
                past_key_value,
                output_attentions,
            )                        
            attention_output = self.output([self_outputs0[0],self_outputs1[0]], hidden_states)
    
            outputs = (attention_output,) + self_outputs0[1:]  # add attentions if we output them
        else:       
            self_outputs = self.self(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            attention_output = self.output(self_outputs[0], hidden_states)
            outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs

    def get_flops(self):
        if hasattr(self, 'self0'):
            flops = self.self0.get_flops() + self.self1.get_flops()
            if hasattr(self.self0, 'alpha'):
                flops += self.output.get_flops(mask=[self.self0.alpha, self.self1.alpha])
            else:
                flops += self.output.get_flops()
        else:
            flops = self.self.get_flops()
            if hasattr(self.self, 'alpha'):
                flops += self.output.get_flops(mask=[self.self.alpha])
            else:
                flops += self.output.get_flops()
        return flops

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        
        self.alpha = nn.Parameter(torch.ones(1, 1, config.intermediate_size))

    def forward(self, hidden_states):
        self.token_num = hidden_states.shape[1]
        hidden_states = self.dense(hidden_states)
        
        if hasattr(self, 'alpha'):
            hidden_states = hidden_states * self.alpha
        
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    def get_flops(self):
        if hasattr(self, 'alpha'):
            flops = self.token_num * self.dense.in_features * self.alpha.sum().item()
        else:
            flops = self.token_num * self.dense.in_features * self.dense.out_features
        return flops
    
    def compress_inter_channels(self, preserve_idxs, template_module):
        in_features = self.dense.in_features
        hidden_features = len(preserve_idxs)
        self.dense = nn.Linear(in_features, hidden_features, bias=True)
        self.dense.weight.data = template_module.dense.weight.data[preserve_idxs, :]
        self.dense.bias.data = template_module.dense.bias.data[preserve_idxs]
        

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        self.token_num = hidden_states.shape[1]
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.densev(self.denseu(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def get_flops(self, mask=None):
        if mask is None:
            in_f = self.dense.in_features
        else:
            in_f = self.dense.in_features / mask.numel() * mask.sum().item()
        flops = (in_f * self.dense.out_features + self.LayerNorm.weight.shape[0] * 5) * self.token_num
        return flops
    
    def compress_inter_channels(self, preserve_idxs, template_module):
        hidden_features = len(preserve_idxs)
        out_features = self.dense.out_features
        self.dense = nn.Linear(hidden_features, out_features, bias=True)
        self.dense.weight.data = template_module.dense.weight.data[:, preserve_idxs]
        self.dense.bias.data = template_module.dense.bias.data


class BertLayer(nn.Module):
    def __init__(self, config, layer_num, attnmask=True):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config, attnmask=attnmask)      
        self.layer_num = layer_num          
        if self.config.add_cross_attention:
            self.crossattention = BertAttention(config, is_cross_attention=self.config.add_cross_attention, layer_num=layer_num, attnmask=attnmask)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.Sparsemax = Sparsemax()

    def Reduce_token(self, x, reduce_num, temperature=0, self_attn=None, cls_attn=None, token_attn=None, mask=None):
        token_num = x.shape[-2]
        if self_attn is not None:
            self_attn_w, head_contri_indices = self_attn[:,:,1:,1:].max(1)  # B, N, N
            self_attn_w = self_attn_w.sum(dim=1)
            self_attn_w = self_attn_w / (self_attn_w.sum(dim=1, keepdim=True) + 1e-8)
        
        if token_attn is not None:
            token_attn_w = token_attn.max(2)[0] # B, N
            token_attn_w = token_attn_w / (token_attn_w.sum(dim=1, keepdim=True) + 1e-8)

        Importance_score = (self_attn_w + token_attn_w + cls_attn) / 3.0
        Importance_score.masked_fill_(mask!=0, 0)
        
        self.importance_score = (Importance_score / Importance_score.sum(dim=1, keepdim=True)).detach()

        if cls_attn is not None and token_attn is not None:
            token_attn /= temperature 
            if mask is not None:
                token_attn = torch.softmax(token_attn+mask.unsqueeze(-1), dim=1).permute(0,2,1) # modified from MADTP: to eliminate the interference of padded tokens
            else:
                token_attn = torch.softmax(token_attn, dim=1).permute(0,2,1)

            score_weight = torch.bmm(token_attn, Importance_score.unsqueeze(-1))
            threshold = torch.min(score_weight, dim=1)[0]
            self.prune_flops = token_attn.shape[1] * token_attn.shape[2]

            idx = Importance_score > threshold

            topk_num = torch.max(idx.sum(dim=1)).item() 
            reduce_num = token_num - topk_num

            if topk_num < 1 or reduce_num <= 1:
                self.top_scores = self.importance_score # for CoMP -- CIP
                return x, mask

            # Merge Token
            _, indices = Importance_score.topk(topk_num, dim=-1, largest=True, sorted=False)
            x_topk = vector_gather(x, indices)
            
            top_scores = vector_gather(Importance_score.unsqueeze(-1), indices).squeeze(-1)
            self.top_scores = (top_scores / top_scores.sum(dim=1, keepdim=True)).detach() # for CoMP -- CIP
            
            Importance_score_weight, indices_sort = Importance_score.topk(token_num, dim=-1, largest=True, sorted=True)
            x_sort = vector_gather(x, indices_sort)
            x_reduce = x_sort[:,topk_num:,:]
            Importance_score_weight = Importance_score_weight[:,topk_num:]
            Importance_score_weight = Importance_score_weight / ( Importance_score_weight.sum(dim=1, keepdim=True) + 1e-8)
            x_combine = torch.bmm(Importance_score_weight.unsqueeze(1), x_reduce)
            self.prune_flops += x_reduce.shape[-1] * x_reduce.shape[-2]
            x = torch.cat([x_topk, x_combine], dim=1)

            if mask is not None:
                new_mask = vector_gather(mask.unsqueeze(2), indices).squeeze(2) # This is relatively simple for encoder / encoder-decoder architecture
                mask = torch.cat([new_mask, torch.gather(mask, dim=-1, index=indices_sort[:,topk_num:]).max(-1)[0].unsqueeze(-1)], dim=1) # if there is a token not masked, the combined token should be un-masked

        return x, mask

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        space_dict=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        mode=None,
        token_attn=None,
        reduce_num=0,
        temperature=0,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            space_dict=space_dict,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        ## get attention value
        self_attn = self.attention.self.get_attention_map()
        cls_attn = self.attention.self.get_cls_attn()
        self.prune_flops = 0.0
        ## Adaptive Reduce Token, inherited from MADTP
        if temperature > 0:
            eos_ft = attention_output[:,:1,:]
            eos_mask = attention_mask[:,:,:,:1]
            patch_ft = attention_output[:,1:,:]
            patch_mask = attention_mask[:,:,:,1:].squeeze(1).squeeze(1)

            patch_ft, patch_mask = self.Reduce_token(patch_ft, reduce_num,temperature, self_attn=self_attn, cls_attn=cls_attn, token_attn=token_attn, mask=patch_mask)

            attention_output = torch.cat([eos_ft, patch_ft], dim=1)
            patch_mask = patch_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = torch.cat([eos_mask, patch_mask], dim=-1)

        if mode=='multimodal':
            assert encoder_hidden_states is not None, "encoder_hidden_states must be given for cross-attention layers"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights                               
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value, attention_mask) 

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def get_flops(self):
        flops = self.attention.get_flops()
        if hasattr(self, 'crossattention'):
            flops += self.crossattention.get_flops()
        flops += self.intermediate.get_flops()
        if hasattr(self.intermediate, 'alpha'):
            flops += self.output.get_flops(mask=self.intermediate.alpha)
        else:
            flops += self.output.get_flops()
        flops += self.prune_flops
        return flops

class BertEncoder(nn.Module):
    def __init__(self, config, sd_dim=768, attnmask=True):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config,i, attnmask=attnmask) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.txt_query_model = Query_model(ft_dim=config.hidden_size, sd_dim=sd_dim, temperature=1, att_func_type='sparsemax', pool_type='max') 

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        space_dict=None,
        temperature=0,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        mode='multimodal',
    ):
        self.querymodel_token_num = 0
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        token_num = hidden_states.shape[-2]
        reduce_num = int((token_num - 1) // self.config.num_hidden_layers)
        sd_txt_ft_all = None      

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            #### reduce token #####
            patch_ft = hidden_states[:,1:,:]
            patch_mask = attention_mask[:,:,:,1:].squeeze(1).squeeze(1)
            self.querymodel_token_num += patch_ft.shape[1]
            token_attn, sd_txt_ft, _ = self.txt_query_model(patch_ft, space_dict, mask=patch_mask, return_token_att=True, temperature=temperature)

            if sd_txt_ft_all is None:
                sd_txt_ft_all = sd_txt_ft
            else:
                sd_txt_ft_all += sd_txt_ft

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    mode=mode,
                    space_dict=space_dict,
                    token_attn=token_attn,
                    reduce_num=reduce_num,
                    temperature=temperature,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    space_dict,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    mode=mode,
                    token_attn=token_attn,
                    reduce_num=reduce_num,
                    temperature=temperature,
                )

            hidden_states = layer_outputs[0]
            attention_mask = layer_outputs[-1]

            if use_cache:
                next_decoder_cache += (layer_outputs[-2],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        ), sd_txt_ft_all

    def get_flops(self):
        flops = 0.0
        for i, layer in enumerate(self.layer):
            flops += layer.get_flops()
        flops += self.querymodel_token_num * self.txt_query_model.token_num * self.txt_query_model.att_dim
        return flops

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
   

class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True, sd_dim=768, attnmask=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        
        self.encoder = BertEncoder(config, sd_dim, attnmask=attnmask)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        if not config.evaluate:
            self.init_weights()
 
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    
    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device, is_decoder: bool) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape

                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
   
                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                            causal_mask,
                        ],
                        axis=-1,
                    )                     

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        space_dict=None,
        temperature=0,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
        mode='multimodal',
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif encoder_embeds is not None:    
            input_shape = encoder_embeds.size()[:-1]
            batch_size, seq_length = input_shape 
            device = encoder_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or encoder_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
            
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, 
                                                                                 device, is_decoder)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            
            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:    
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        if encoder_embeds is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
        else:
            embedding_output = encoder_embeds
            
        encoder_outputs, sd_txt_ft = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            space_dict=space_dict,
            temperature=temperature,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode=mode,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        ), sd_txt_ft

    def get_flops(self):
        flops = self.embeddings.get_flops()
        flops += self.encoder.get_flops()
        return flops