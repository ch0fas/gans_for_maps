import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

# Directory
work_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(work_dir, "../data/raw/satellite/")

# Create directory
os.makedirs(data_dir, exist_ok=True)

# Transformation: from PIL (image) to Tensor
transform = transforms.Compose([
    transforms.ToTensor(),  
])

# Load EuroSAT RGB dataset
dataset = datasets.EuroSAT(
    root="data",                    # Folder to download ENTIRE dataset
    transform=transform,
    download=True
)

# Filter images from dataset. Keep certain images from certain classes
classes_to_include = ["Forest", "Pasture", "Residential", "River", "Highway", "PermanentCrop"]
class_to_idx = {v: k for k, v in dataset.class_to_idx.items()}

# Max images per class
max_per_class = 100  

# Store images based on index
selected_indices = []
class_counts = {c: 0 for c in classes_to_include}

for i, (_, label) in enumerate(dataset):
    class_name = dataset.classes[label]
    if class_name in classes_to_include and class_counts[class_name] < max_per_class:
        selected_indices.append(i)
        class_counts[class_name] += 1

# Save images in the same folder
for i, idx in enumerate(selected_indices):
    img, _ = dataset[idx]
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(data_dir, f"tile_{i:05d}.png"))

print(f"Saved {len(selected_indices)} images to {data_dir}")
print("Images per class:", class_counts)