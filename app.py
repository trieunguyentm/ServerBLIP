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

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'
    
model = blip_decoder(pretrained=model_url, image_size=384, vit='large')
model.eval()
model = model.to(device)


from models.blip_vqa import blip_vqa

image_size_vq = 480
transform_vq = transforms.Compose([
    transforms.Resize((image_size_vq,image_size_vq),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 

model_url_vq = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_vqa.pth'
    
model_vq = blip_vqa(pretrained=model_url_vq, image_size=480, vit='base')
model_vq.eval()
model_vq = model_vq.to(device)



def inference(raw_image, model_n, question, strategy):
    if model_n == 'Image Captioning':
        image = transform(raw_image).unsqueeze(0).to(device)   
        with torch.no_grad():
          if strategy == "beam search":
              caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
          else:
              caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
          return 'caption: '+caption[0]

    else:   
        image_vq = transform_vq(raw_image).unsqueeze(0).to(device)  
        with torch.no_grad():
            answer = model_vq(image_vq, question, train=False, inference='generate') 
        return  'answer: '+answer[0]
    
inputs = [gr.inputs.Image(type='pil'),gr.inputs.Radio(choices=['Image Captioning',"Visual Question Answering"], type="value", default="Image Captioning", label="Model"),"textbox",gr.inputs.Radio(choices=['Beam search','Nucleus sampling'], type="value", default="Nucleus sampling", label="Caption Decoding Strategy")]
outputs = gr.outputs.Textbox(label="Output")

title = "BLIP"

description = "Gradio demo for BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (Salesforce Research). To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."

article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2201.12086' target='_blank'>BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation</a> | <a href='https://github.com/salesforce/BLIP' target='_blank'>Github Repo</a></p>"


gr.Interface(inference, inputs, outputs, title=title, description=description, article=article, examples=[['starrynight.jpeg',"Image Captioning","None","Nucleus sampling"]]).launch(enable_queue=True)