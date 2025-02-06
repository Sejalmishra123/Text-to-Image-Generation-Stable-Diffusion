!pip install diffusers transformers accelerate

from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# Load the pre-trained Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

def generate_image(prompt):
    """Generates an image based on a text prompt."""
    image = pipe(prompt, num_inference_steps=50).images[0]
    return image

# Example usage
prompt = "A majestic red dragon soaring through a starry night sky"
generated_image = generate_image(prompt)

# Display the generated image (using matplotlib)
plt.imshow(generated_image)
plt.show()
