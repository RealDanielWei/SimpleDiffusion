from StableDiffusionSystem import StableDiffusionConfig, StableDiffusionSystem
import gradio as gr

sd_config = StableDiffusionConfig()
sd = StableDiffusionSystem(sd_config)

def set_seed(seed):
    sd.generator.manual_seed(seed)

def text2img_gradio(prompt, negative_prompt, steps, progress=gr.Progress(track_tqdm=True)):
    return sd.textToImage(prompt=prompt, negative_prompt = negative_prompt, steps = steps)[0]

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

with gr.Blocks() as demo:
    gr.Markdown("SimpleDiffusion UI")
    model_name = gr.Dropdown(value = sd.config.model_name, choices = [sd.config.model_name], label= "Model")
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
        image_to_generate.change(text2img_gradio, inputs=[prompt, negative_prompt, steps], outputs = img_out).then(transfer, inputs = img_out, outputs = gallery).then(count_down_to_zero, inputs = image_to_generate, outputs = image_to_generate)

    with gr.Tab("Flip Image"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("Flip")

demo.queue().launch()