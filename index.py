import os
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline

# 1. Load the pipeline
pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)

# 2. Load your input image
image = Image.open("./purse.png").convert("RGB")

# 3. Craft a very specific prompt
prompt = (
    "Replace the main body material of the purse with red leather texture, "
    "keep the shape, stitching, and hardware unchanged, "
    "apply a rich red color with a smooth leather grain, realistic, high quality."
)

# 4. Set generation parameters
generator = torch.manual_seed(42)  # change seed for variation
true_cfg_scale = 7.0              # higher = stronger prompt adherence
num_inference_steps = 40          # 40 gives good quality, lower steps for speed

inputs = {
    "image": image,
    "prompt": prompt,
    "generator": generator,
    "true_cfg_scale": true_cfg_scale,
    "negative_prompt": "blurry, distorted, deformed, wrong color, low quality",
    "num_inference_steps": num_inference_steps,
}

# 5. Generate and save
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("red_leather_purse.png")
    print("Image saved as red_leather_purse.png")
