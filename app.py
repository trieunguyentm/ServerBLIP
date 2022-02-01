import os
os.system("wget https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1920px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg -O starry.jpg")

from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





import gradio as gr

from models.blip import blip_decoder

image_size = 384
transform = transforms.Compose([
    transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
    
model = blip_decoder(pretrained=model_url, image_size=384, vit='base')
model.eval()
model = model.to(device)

    
def inference(raw_image):
    image = transform(raw_image).unsqueeze(0).to(device)     
    with torch.no_grad():
      caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
      print('caption: '+caption[0])

    return 'caption: '+caption[0]
    
inputs = gr.inputs.Image(type='pil')
outputs = gr.outputs.Textbox(label="Output")

title = "BLIP"

description = "Gradio demo for BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."

article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2201.12086' target='_blank'>BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation</a> | <a href='https://github.com/salesforce/BLIP' target='_blank'>Github Repo</a></p>"


gr.Interface(inference, inputs, outputs, title=title, description=description, article=article, examples=[['starry.jpg']]).launch(enable_queue=True,cache_examples=True)