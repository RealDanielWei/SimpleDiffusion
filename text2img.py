from StableDiffusionSystem import *
import argparse

parser = argparse.ArgumentParser(description='Standalone commandline tool for text to image tasks')
parser.add_argument('-p', '--prompt', default= "a photograph of an astronaut riding a horse.", help='input prompt')
parser.add_argument('-o', '--output', default= "outImg", help='output file name')
parser.add_argument('-d', '--device', default= "cpu", help = 'device to run the task')
parser.add_argument('-m', '--model', default= "runwayml/stable-diffusion-v1-5", help = 'model name')
parser.add_argument('-s', '--steps', default= 25, help='inference steps')
parser.add_argument('-g', '--guidance', default= 7.5, help= 'guidance scale')

# Parse the command-line arguments
args = parser.parse_args()

#Change default config if requested
sd_config = StableDiffusionConfig()
sd_config.device = args.device
sd_config.model_name = args.model 
sd_config.num_inference_steps = int(args.steps)
sd_config.guidance_scale = float(args.guidance) 
#Prepare sd system 
sd = StableDiffusionSystem(sd_config)
#Perform inference
image = sd.textToImage([args.prompt])
#Save output
pil_image = ToPILImage(image)
for id in range(len(pil_image)):
        pil_image[id].save(args.output + "_" + str(id) + ".jpg", format="JPEG")


