from StableDiffusionSystem import StableDiffusionConfig, StableDiffusionSystem
import gradio as gr

sd_config = StableDiffusionConfig()
sd = StableDiffusionSystem(sd_config)

def set_seed(seed):
    sd.generator.manual_seed(seed)

def text2img_gradio(model_name, prompt, negative_prompt, steps, progress=gr.Progress(track_tqdm=True)):
    return sd.textToImage(model_name = model_name, prompt = prompt, negative_prompt = negative_prompt, steps = steps)[0]

img_list = []
def transfer(img):
    img_list.append(img)
    return img_list

def minus_one(x):
    return int(x) - 1 

def count_down_to_zero(x):
    if(int(x) > 0):
        return int(x) - 1
    return int(x)

def getMask(img : gr.ImageMask):
    return img["mask"]

def inpainting_gradio(model_name, prompt, negative_prompt, img, steps, progress=gr.Progress(track_tqdm=True)):
    return sd.inpainting(model_name = model_name, prompt = prompt, negative_prompt = negative_prompt, input_img = img["image"], mask_img = img["mask"], steps = steps)[0]

with gr.Blocks() as demo:
    gr.Markdown("SimpleDiffusion UI")
    model_name = gr.Dropdown(value = "runwayml/stable-diffusion-v1-5", choices = ["runwayml/stable-diffusion-v1-5", "runwayml/stable-diffusion-inpainting"], label= "Model")
    with gr.Tab("Text2Image"):
        prompt = gr.Textbox(label= "Positive Prompt")
        negative_prompt = gr.Textbox(label= "Negative Prompt")
        steps = gr.Slider(label= "Number of Steps", minimum = 1, maximum = 30, value = 20, step = 1)
        seed = gr.Slider(label= "Random seed", minimum = 1, maximum = 2147483647, value = 666, step = 1)
        total_requested_image_num = gr.Number(label="Number of images to generate", minimum = 0, maximum = 300, value = 0, step = 1)
        image_to_generate = gr.Number(visible = False)
        generate_button = gr.Button(value = "Generate")
        img_out = gr.Image(label = "Output")
        gallery = gr.Gallery()
        clear_button = gr.ClearButton([gallery])
        generate_button.click(set_seed, inputs = seed, outputs = None).then(minus_one, inputs = total_requested_image_num, outputs = image_to_generate)
        image_to_generate.change(text2img_gradio, inputs=[model_name, prompt, negative_prompt, steps], outputs = img_out).then(transfer, inputs = img_out, outputs = gallery).then(count_down_to_zero, inputs = image_to_generate, outputs = image_to_generate)

    with gr.Tab("Inpainting"):
        with gr.Row():
            image_input = gr.ImageMask()
            image_output = gr.Image()
        prompt = gr.Textbox(label= "Prompt")
        negative_prompt = gr.Textbox(label= "Negative Prompt")
        steps = gr.Slider(label= "Number of Steps", minimum = 1, maximum = 30, value = 20, step = 1)
        seed = gr.Slider(label= "Random seed", minimum = 1, maximum = 2147483647, value = 666, step = 1)
        inpaint_button = gr.Button("Inpaint")
        inpaint_button.click(set_seed, inputs = seed, outputs = None).then(inpainting_gradio, inputs = [model_name, prompt, negative_prompt, image_input, steps], outputs = image_output)

demo.queue().launch()