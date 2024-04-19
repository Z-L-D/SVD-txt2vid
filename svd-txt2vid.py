import torch

from txt2vid_pipeline import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "C:\Production\Applied Science\Software\LocalSD\Models\stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Define the text input for conditioning
text_input = "A futuristic cityscape at sunset with flying cars and neon signs."

# Set the random seed for reproducibility
generator = torch.manual_seed(42)

# Generate frames using the text input
frames = pipe(text=text_input, decode_chunk_size=8, generator=generator).frames[0]

# Export the generated frames to a video
from diffusers.utils import export_to_video
export_to_video(frames, "generated_text_to_video.mp4", fps=7)