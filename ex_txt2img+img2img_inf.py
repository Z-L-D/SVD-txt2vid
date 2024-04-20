import torch
from diffusers import StableCascadeCombinedPipeline

pipe = StableCascadeCombinedPipeline.from_pretrained(
    "stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
prompt = "an image of a shiba inu, donning a spacesuit and helmet"
images = pipe(prompt=prompt)