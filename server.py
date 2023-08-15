from flask import Flask, request, jsonify
import requests
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from io import BytesIO

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

from models.blip import blip_decoder
from models.blip_vqa import blip_vqa

image_size = 384
transform = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
]) 
model_url = "model_large_caption.pth"
model = blip_decoder(pretrained=model_url, image_size=384, vit='large')
model.eval()
model = model.to(device)

image_size_vq = 480
transform_vq = transforms.Compose([
    transforms.Resize((image_size_vq, image_size_vq), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
]) 

model_url_vq = 'model__vqa.pth'
model_vq = blip_vqa(pretrained=model_url_vq, image_size=480, vit='base')
model_vq.eval()
model_vq = model_vq.to(device)

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = image.convert("RGB")  # Convert to RGB mode
    return image

def inference(raw_image, model_n, question, strategy):
    if model_n == 'Image Captioning':
        image = transform(raw_image).unsqueeze(0).to(device)   
        with torch.no_grad():
            if strategy == "Beam search":
                caption = model.generate(image, sample=False, num_beams=3, max_length=30, min_length=5)
            else:
                caption = model.generate(image, sample=True, top_p=0.9, max_length=30, min_length=5)
            return 'caption: ' + caption[0]
    else:   
        image_vq = transform_vq(raw_image).unsqueeze(0).to(device)  
        with torch.no_grad():
            answer = model_vq(image_vq, question, train=False, inference='generate') 
        return 'answer: ' + answer[0]

@app.route('/openai/text/blip', methods=['POST'])
def blip_caption():
    data = request.json
    image_url = data['image_url']
    model_name = data['model_name']
    question = data['question']
    caption_strategy = data['caption_strategy']

    image = load_image_from_url(image_url)

    result = inference(image, model_name, question, caption_strategy)

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(port=9000)
