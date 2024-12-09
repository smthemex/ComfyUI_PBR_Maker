# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import random

import torch
from PIL import Image
import numpy as np
from diffusers import DiffusionPipeline
import folder_paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor_to_pil(tensor):
    image = Image.fromarray(tensor.squeeze().mul(255).clamp(0, 255).byte().numpy(), mode='RGB')
    return image

def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img
class Load_MatForger:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING",{"default": "gvecchio/MatForger"}),
            }
        }

    RETURN_TYPES = ("MatForgerMODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_matforger"
    CATEGORY = "MatForger"

    def load_matforger(self, repo_id,):
       
        if not repo_id :
            raise "need fill repo_id or loacl repo "
        
        pipe = DiffusionPipeline.from_pretrained(repo_id,trust_remote_code=True)
        pipe.enable_vae_tiling()
        pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)
        return (pipe,)
class MatForger_Sampler:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MatForgerMODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "terracotta brick wall"}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 1, "max": 100, "step": 0.1,  "display": "number"}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "display": "number"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "display": "number"}),
                "step": ("INT", {"default": 25, "min": 1, "max": 4096, "step": 1, "display": "number"}),
                "tileable": ("BOOLEAN", {"default": True},),
                "patched": ("BOOLEAN", {"default": False},),
                "Save_with_prefix": ("BOOLEAN", {"default": False},),
            },
            "optional":{"image": ("IMAGE",),
                        }
        }

    RETURN_TYPES = ("IMAGE","IMAGE")
    RETURN_NAMES = ("image","image_l")
    FUNCTION = "txt2img_sampler"
    CATEGORY = "MatForger"

    def txt2img_sampler(self,pipe,prompt,cfg,height,width,step,tileable,patched,Save_with_prefix,**kwargs):
        pipe.to(device)
        prompts = prompt.splitlines()
        image=kwargs.get("image")
        if isinstance(image,torch.Tensor):
            batch_size=image.shape[0]
            if batch_size==1:
                prompts = [tensor_to_pil(image)]
            else:
                prompts = [tensor_to_pil(img) for img in list(torch.chunk(image, chunks=batch_size))]
                
        image_RGB_list=[]
        image_L_list = []
        for prompt in prompts:
            image = pipe(
                prompt,
                guidance_scale=cfg,
                height=height,
                width=width,
                tileable=tileable,
                patched=patched,
                num_inference_steps=step,
                
            ).images[0]
            prefix= ''.join(random.choice("0123456789abcdefg") for _ in range(6))
            if Save_with_prefix:
                print(f"save mat in name {prefix}")
                image.basecolor.save(folder_paths.get_input_directory(),f"basecolor_{prefix}")
                image.normal.save(folder_paths.get_input_directory(),f"normal{prefix}")
                image.height.save(folder_paths.get_input_directory(),f"height{prefix}")
                image.roughness.save(folder_paths.get_input_directory(),f"roughness{prefix}")
                image.metallic.save(folder_paths.get_input_directory(),f"metallic{prefix}")
                
            image_RGB_list.append([image.basecolor, image.normal])
            image_L_list.append([image.height, image.roughness, image.metallic])
            
        #print(image_L_list,image_RGB_list)
        RGB_value_list=[]
        for i in image_RGB_list:
            for j in i:
                RGB_value_list.append(j)
                
        image_L_alue = []
        for i in image_L_list:
            for j in i:
                image_L_alue.append(j)
        
        RGB_value_list=[phi2narry(i) for i in RGB_value_list]
        image_L_alue = [phi2narry(i) for i in image_L_alue]
        
        image = torch.from_numpy(np.fromiter(RGB_value_list, np.dtype((np.float32, (height, width, 3)))))
        image_l= torch.from_numpy(np.fromiter(image_L_alue, np.dtype((np.float32, (height, width)))))
        return (image,image_l)


NODE_CLASS_MAPPINGS = {
    "Load_MatForger":Load_MatForger,
    "MatForger_Sampler": MatForger_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load_MatForger":"Load_MatForger",
    "MatForger_Sampler": "MatForger_Sampler",
}
