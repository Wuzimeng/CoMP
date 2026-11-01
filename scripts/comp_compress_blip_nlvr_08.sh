#!/bin/bash
save_path=output/comp_compress_blip_nlvr_08
mkdir $save_path

python -m torch.distributed.run --nproc_per_node=2 --master_port 11463 comp_compress_blip_nlvr.py --epoch 30 \
--p 0.8 \
--pretrained pretrained/blip/model_base_nlvr.pth \
--config ./configs/nlvr.yaml \
--output_dir $save_path \
2>&1 | tee $save_path/train.log