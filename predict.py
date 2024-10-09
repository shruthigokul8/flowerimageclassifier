import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import json
import argparse

def get_input_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    parser.add_argument("image_path", type=str, help="Path to the image")
    parser.add_argument("checkpoint", type=str, help="Model checkpoint to load")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="JSON file with category names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['model'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    image = Image.open(image_path)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return preprocess(image).unsqueeze(0)

def predict(image_path, model, top_k=5, gpu=False):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Process image
    image = process_image(image_path).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(top_k, dim=1)
        
        # Invert class_to_idx
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        top_class = [idx_to_class[i] for i in top_class.cpu().numpy()[0]]
        
    return top_p.cpu().numpy()[0], top_class

if __name__ == "__main__":
    args = get_input_args()
    
    # Load the model
    model = load_checkpoint(args.checkpoint)
    
    # Load the category names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Make prediction
    probs, classes = predict(args.image_path, model, args.top_k, args.gpu)
    
    # Print the results
    flower_names = [cat_to_name[str(cls)] for cls in classes]
    for prob, flower_name in zip(probs, flower_names):
        print(f"{flower_name}: {prob:.4f}")
