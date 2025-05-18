## Prototype Development for Image Generation Using the Stable Diffusion Model and Gradio Framework

### AIM:
To design and deploy a prototype application for image generation utilizing the Stable Diffusion model, integrated with the Gradio UI framework for interactive user engagement and evaluation.

### PROBLEM STATEMENT:
To develop an interactive application that allows users to generate custom images from text prompts using a state-of-the-art text-to-image model. The system should be user-friendly, enabling efficient testing and evaluation of the model's capabilities through a Gradio-based interface.
### DESIGN STEPS:

#### STEP 1: Understand the API and Framework Requirements 
- Identify the Stable Diffusion model's inference endpoint.  
- Set up authentication using Hugging Face API keys and configure environment variables.
  
#### STEP 2: Develop Backend Logic 
- Implement API request logic to send prompts and receive image outputs.  
- Convert API responses (base64-encoded images) into displayable image formats.  

#### STEP 3: Design Gradio Interface 
- Build an interactive Gradio interface to accept prompts and display images.  
- Add labels, descriptions, and examples for a user-friendly experience.

### PROGRAM:
```py
# ---------------------------------------------
# 1. Import Required Libraries
# ---------------------------------------------
import os
import io
import base64
import json
import requests
from PIL import Image
import IPython.display
from dotenv import load_dotenv, find_dotenv
import gradio as gr

# ---------------------------------------------
# 2. Load Environment Variables (API Keys, URLs)
# ---------------------------------------------
_ = load_dotenv(find_dotenv())  # Load .env file
hf_api_key = os.environ['HF_API_KEY']

# ---------------------------------------------
# 3. Helper Function to Call Hugging Face Text-to-Image API
# ---------------------------------------------
def get_completion(inputs, parameters=None, ENDPOINT_URL=os.environ['HF_API_TTI_BASE']):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters is not None:
        data.update({"parameters": parameters})

    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(data))
    return json.loads(response.content.decode("utf-8"))

# ---------------------------------------------
# 4. Convert Base64 to PIL Image
# ---------------------------------------------
def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = io.BytesIO(base64_decoded)
    pil_image = Image.open(byte_stream)
    return pil_image

# ---------------------------------------------
# 5. Basic Image Generator Function (Gradio Interface #1)
# ---------------------------------------------
def generate(prompt):
    output = get_completion(prompt)
    result_image = base64_to_pil(output)
    return result_image

gr.close_all()
demo = gr.Interface(
    fn=generate,
    inputs=[gr.Textbox(label="Your prompt")],
    outputs=[gr.Image(label="Result")],
    title="Image Generation with Stable Diffusion",
    description="Generate any image with Stable Diffusion",
    allow_flagging="never",
    examples=["a student going to his school in city of Chennai"]
)
demo.launch(share=True, server_port=int(os.environ['PORT1']))
demo.close()

# ---------------------------------------------
# 6. Advanced Generator with Prompt Customization (Gradio Interface #2)
# ---------------------------------------------
def generate(prompt, negative_prompt, steps, guidance, width, height):
    params = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height
    }
    output = get_completion(prompt, params)
    return base64_to_pil(output)

gr.close_all()
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Your prompt"),
        gr.Textbox(label="Negative prompt"),
        gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25),
        gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7),
        gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512),
        gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512)
    ],
    outputs=[gr.Image(label="Result")],
    title="Image Generation with Stable Diffusion",
    description="Generate any image with Stable Diffusion",
    allow_flagging="never"
)
demo.launch()
demo.close()

# ---------------------------------------------
# 7. Clean Layout with Gr.Blocks() (Gradio Interface #3)
# ---------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Image Generation with Stable Diffusion")
    prompt = gr.Textbox(label="Your prompt")
    with gr.Row():
        with gr.Column():
            negative_prompt = gr.Textbox(label="Negative prompt")
            steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25)
            guidance = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7)
            width = gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512)
            height = gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512)
            btn = gr.Button("Submit")
        with gr.Column():
            output = gr.Image(label="Result")

    btn.click(fn=generate, inputs=[prompt, negative_prompt, steps, guidance, width, height], outputs=[output])

gr.close_all()
demo.launch()
demo.close()

# ---------------------------------------------
# 8. Responsive UI with Accordion for Advanced Settings (Gradio Interface #4)
# ---------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Image Generation with Stable Diffusion")
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Your prompt")
        with gr.Column(scale=1, min_width=50):
            btn = gr.Button("Submit")

    with gr.Accordion("Advanced options", open=False):
        negative_prompt = gr.Textbox(label="Negative prompt")
        with gr.Row():
            with gr.Column():
                steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25)
                guidance = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7)
            with gr.Column():
                width = gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512)
                height = gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512)

    output = gr.Image(label="Result")
    btn.click(fn=generate, inputs=[prompt, negative_prompt, steps, guidance, width, height], outputs=[output])

gr.close_all()
demo.launch()
gr.close_all()
```
### OUTPUT:
# 1. Output
![Output Experiment 7(1) Gen AI](https://github.com/user-attachments/assets/f978f394-9429-4d77-ba82-cc3eb8d54880)
# 2. Output
![Ouput Experiment 7 (2) Gen AI](https://github.com/user-attachments/assets/fb8e98af-c129-47c8-a29c-b51b03d1f6b4)
# 3. Output
![Output Experiment 7(3) Gen AI](https://github.com/user-attachments/assets/f51ac7c8-362e-4555-86a1-ccd621b58723)
# 4. Output
![Output Experiment 7(4) Gen AI](https://github.com/user-attachments/assets/da9a43c1-f11b-4642-93ca-95c7c1d6ae25)
# 5. Output
![Output Experiment 7(5)(1) Gen AI](https://github.com/user-attachments/assets/e82401b8-de3e-4d03-9648-604cb59bd426)
![Output Experiment 7(5)(2) Gen AI](https://github.com/user-attachments/assets/52b357af-b939-4971-8947-9bae1282a4c1)

### RESULT:
Successfully designed and deployed a prototype application for image generation using the Stable Diffusion model, demonstrating interactive user engagement and the ability to generate high-quality images from text prompts.
