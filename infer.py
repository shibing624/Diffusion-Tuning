# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import torch
from diffusers import AltDiffusionPipeline, UNet2DConditionModel


def main():
    device = "cuda:0"
    alt_model_dir = "BAAI/AltDiffusion-m18"
    pipe = AltDiffusionPipeline.from_pretrained(alt_model_dir, torch_dtype=torch.float16, device_map={"": device})
    prompt = "yoda"
    image = pipe(prompt).images[0]
    print(image)

    ft_model_path = "outputs/sd-v1/checkpoint-9000/unet"
    unet = UNet2DConditionModel.from_pretrained(ft_model_path, torch_dtype=torch.float32, device_map={"": device})
    pipe = AltDiffusionPipeline.from_pretrained(alt_model_dir, unet=unet, torch_dtype=torch.float32,
                                                safety_checker=None,
                                                device_map={"": device})

    prompt = "yoda"
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, width=512, height=512, ).images[0]
    print(image)


if __name__ == '__main__':
    main()
