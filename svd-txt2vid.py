import torch

from pipeline_mod import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from transformers import AutoTokenizer, T5EncoderModel
import argparse

# Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--sd_dir", type=str, default="")
    # parser.add_argument("--scheduler", type=str, default="")
    parser.add_argument("--t5_dir", type=str, default="")
    args = parser.parse_args()
    return args

def main(args, prompts):
    torch_device = "cuda"
    
    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained("E:\Applications\LocalGPT\Models\T5")
    text_encoder = T5EncoderModel.from_pretrained("E:\Applications\LocalGPT\Models\T5").to(torch_device)
    text_encoder.eval()

    # Load Stable Diffusion
    ##############
    #################
    
    # Inference
    with torch.no_grad():
        print("\n=== DEBUG START ================")
        print("=== prompt START =============")
        print("Warning: One or more processed inputs are None.")
        # Inspect inputs to identify the problem
        print(prompt)
        print("=== DEBUG END ==================\n")

        text_ids = tokenizer(prompt, padding="max_length", max_length=308, return_tensors="pt", truncation=True).input_ids.to(torch_device)

        print("\n=== DEBUG START ================")
        print("=== text_ids START =============")
        print("Warning: One or more processed inputs are None.")
        # Inspect inputs to identify the problem
        print(text_ids)
        print("=== DEBUG END ==================\n")

        # # Text embeddings
        # text_embeddings = text_encoder(input_ids=text_ids, output_hidden_states=True).hidden_states[-1].to(torch.float32)
        # print("\n=== DEBUG START =============")
        # print("=== text_embeddings =============")
        # print("Warning: One or more processed inputs are None.")
        # # Inspect inputs to identify the problem
        # print(text_embeddings)
        # print("=== DEBUG END ===================\n")

        # text_embeddings = adapter(text_embeddings).sample
        # uncond_input = tokenizer([""], padding="max_length", max_length=308, return_tensors="pt")
    
        # pipe = StableVideoDiffusionPipeline.from_pretrained(
        #   "E:/Applications/LocalSD/Models/svd", torch_dtype=torch.float16, variant="fp16"
        # )
        # pipe.enable_model_cpu_offload()
        
        # # Load the conditioning image
        # image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
        # image = image.resize((1024, 576))
        
        # generator = torch.manual_seed(42)
        # frames = pipe(image, num_frames=7, decode_chunk_size=8, generator=generator, motion_bucket_id=180, noise_aug_strength=0.1).frames[0]
        # export_to_video(frames, "generated.mp4", fps=7)

if __name__ == "__main__":
    prompt = "This image captures a female performer on stage, her profile cast in a dramatic chiaroscuro effect that accentuates the contours and highlights of her face and body. She stands with a microphone in hand, her posture relaxed yet exuding confidence and poise, her head slightly tilted upward, eyes gently closed, as if savoring the emotions of the song she’s delivering. The ambient lighting lends an intense mood to the scene, with warm and cool tones blending in the backdrop—splashes of magenta, teal, and golden hues dance across the setting, suggesting a vibrant atmosphere typical of a live music venue. The subject's attire is a sleek, sleeveless top that glistens with perspiration, suggesting the physical intensity of her performance. Strands of her hair, tousled and free-flowing, catch the stage lights, creating a halo effect around her silhouette. Her expression is one of profound expression, the arch of her brows and slight parting of her lips imparting an air of deep concentration or emotional outpour. In the background, the suggestion of musical equipment, perhaps the gleam of cymbals and the silhouette of a drum set, anchors the scene within the context of a concert. The lights converge on her, making her the unequivocal focal point and casting dramatic shadows that amplify the theatricality of the moment. The image encapsulates the essence of a live music performance, where the energy of the audience and the passion of the artist coalesce into an ephemeral, yet memorable tableau."
    args = parse_args()
    main(args, prompt)