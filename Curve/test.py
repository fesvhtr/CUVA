from transformers import Blip2Processor
from PIL import Image

if __name__ == '__main__':
    img_path = r"/home/dh/pythonProject/AnomalyDataset/Data/capture_image/100.jpg"
    processor = Blip2Processor.from_pretrained(r'/home/dh/zjy/ChatCaptioner-main/Video_ChatCaptioner/blip2-flan-t5-xl')
    img = Image.open(img_path)
    print(img)
    res = processor(images=img, return_tensors='pt')