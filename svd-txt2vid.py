# import torch

# from txt2vid_pipeline import StableVideoDiffusionPipeline
# from diffusers.utils import export_to_video

# pipe = StableVideoDiffusionPipeline.from_pretrained(
#     "E:\Applications\LocalSD\Models\diffusers\models--stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16, variant="fp16"
# )
# pipe.enable_model_cpu_offload()

# # Define the text input for conditioning
# text_input = "A futuristic cityscape at sunset with flying cars and neon signs."

# # Set the random seed for reproducibility
# generator = torch.manual_seed(42)

# # Generate frames using the text input
# frames = pipe(text=text_input, decode_chunk_size=8, generator=generator).frames[0]

# # Export the generated frames to a video
# from diffusers.utils import export_to_video
# export_to_video(frames, "generated_text_to_video.mp4", fps=7)

# import torch
# import PIL.Image

# from txt2vid_pipeline import StableVideoDiffusionPipeline
# from diffusers.utils import export_to_video
# from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection, CLIPImageProcessor
# from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel, EulerDiscreteScheduler

# pipe = StableVideoDiffusionPipeline(
#     tokenizer=CLIPTokenizer.from_pretrained("E:/Applications/LocalSD/Models/diffusers/models--stable-video-diffusion-img2vid-xt-1-1/tokenizer"),
#     text_encoder=CLIPTextModel.from_pretrained("E:/Applications/LocalSD/Models/diffusers/models--stable-video-diffusion-img2vid-xt-1-1/text_encoder"),
#     vae=AutoencoderKLTemporalDecoder.from_pretrained("E:/Applications/LocalSD/Models/diffusers/models--stable-video-diffusion-img2vid-xt-1-1/vae"),
#     image_encoder=CLIPVisionModelWithProjection.from_pretrained("E:/Applications/LocalSD/Models/diffusers/models--stable-video-diffusion-img2vid-xt-1-1/image_encoder"),
#     unet=UNetSpatioTemporalConditionModel.from_pretrained("E:/Applications/LocalSD/Models/diffusers/models--stable-video-diffusion-img2vid-xt-1-1/unet"),
#     scheduler=EulerDiscreteScheduler.from_pretrained("E:/Applications/LocalSD/Models/diffusers/models--stable-video-diffusion-img2vid-xt-1-1/scheduler"),
#     feature_extractor=CLIPImageProcessor.from_pretrained("E:/Applications/LocalSD/Models/diffusers/models--stable-video-diffusion-img2vid-xt-1-1/feature_extractor"),
# )

# pipe.enable_model_cpu_offload()

# # Define the text input for conditioning
# text_input = "A futuristic cityscape at sunset with flying cars and neon signs."

# # Load the image input
# image_input = PIL.Image.open("E:/Applications/LocalSD/DATASETS/00001-2184721316.png")

# # Set the random seed for reproducibility
# generator = torch.manual_seed(42)

# # Generate frames using the text and image inputs
# frames = pipe(prompt=text_input, image=image_input, height=576, width=1024, num_frames=25, decode_chunk_size=8, generator=generator).frames[0]

# # Export the generated frames to a video
# export_to_video(frames, "generated_text_to_video.mp4", fps=7)

import torch
import PIL.Image

from txt2vid_pipeline import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "E:/Applications/LocalSD/Models/diffusers/models--stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")

# Define the text input for conditioning
text_input = "Panning across the scene to the left"

# Load the image input
image = load_image(
    "E:/Applications/LocalSD/DATASETS/00001-2184721316.png"
)
image = image.resize((1024, 576))

frames = pipe(image, prompt=text_input, num_frames=25, decode_chunk_size=8).frames[0]
export_to_video(frames, "generated.mp4", fps=7)