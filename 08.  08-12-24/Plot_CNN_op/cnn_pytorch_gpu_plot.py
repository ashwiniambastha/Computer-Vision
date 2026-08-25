import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from matplotlib import cm

# Create a folder for saving images
os.makedirs('temp', exist_ok=True)

# Load the pretrained VGG16 model and move it to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=True).to(device)
model.eval()

# Define a transform to preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to VGG16 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load an image and preprocess
image_path = "./images/golden-retriever.jpg"  # Replace with your image path
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU


layers_to_process = 3

# Get the first x layers of the VGG16 model
layers = list(model.features)[:layers_to_process]

# Pass the image through the layers and save the feature maps
x = input_tensor
for idx, layer in enumerate(layers):
    x = layer(x)  # Forward pass through the layer
    feature_map = x.detach().cpu().numpy()[0]  # Move data to CPU for saving

    # Normalize feature maps to the range [0, 255]
    feature_map -= feature_map.min()
    feature_map /= feature_map.max()
    feature_map *= 255
    feature_map = feature_map.astype(np.uint8)

    counter = 0

    # Save each filter in the current layer as an image
    for filter_idx in range(feature_map.shape[0]):
        filter_image = Image.fromarray(feature_map[filter_idx])
        filter_image = filter_image.convert('L')  # Convert to grayscale
        
        # Save image with descriptive filename
        save_path = f"temp/layer_{idx + 1}_filter_{filter_idx + 1}.png"
        filter_image.save(save_path)
        if counter < 1:
            print(filter_image.size)
        counter += 1

    print(f"Saved feature maps of layer {idx + 1} to 'temp/' folder.")
