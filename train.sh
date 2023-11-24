export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_NAME="BAAI/AltDiffusion-m18"
DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch --mixed_precision="fp16" --multi_gpu train_text_to_image_ad.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 --use_8bit_adam  --gradient_checkpointing \
  --max_train_steps=25000 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine_with_restarts" --lr_warmup_steps=100 \
  --output_dir="outputs/sd-v1"
