# ComfyUI_PBR_Maker
Use comfyUI make PBR materials..

----

1.Installation  
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_PBR_Maker
```
2.checkpoints 
----
download [gvecchio/MatForger](https://huggingface.co/gvecchio/MatForger) 

```
├── anypath/MatForger
|     ├── prompt_encoder
|           ├── config.json
|           ├── diffusion_pytorch_model.safetensors
|           ├── encoder.py
|     ├── scheduler
|           ├── scheduler_config.json
|     ├──unet
|           ├── config.json
|           ├── diffusion_pytorch_model.safetensors
|     ├──vae
|           ├── config.json
|           ├── diffusion_pytorch_model.safetensors
|     ├──model_index.json
|     ├──pipeline.py
```

3.Example:
----
![](https://github.com/smthemex/ComfyUI_PBR_Maker/blob/main/example.png)

4.Citation
--
using author another project ...
 ``` 
@article{vecchio2024stablematerials,
  title={StableMaterials: Enhancing Diversity in Material Generation via Semi-Supervised Learning},
  author={Vecchio, Giuseppe},
  journal={arXiv preprint arXiv:2406.09293},
  year={2024}
}

```
