import os
import random
import torch
from torchvision import transforms
from PIL import Image
from models import networks
from pytorch_fid import fid_score
import numpy as np
from scipy import linalg


# Config
REAL_DIR = "data/raw/styles_google"
INPUT_DIR = "data/raw/sentinel"
REAL_RESIZED = "data/processed/real_resized_sentinel"
FAKE_DIR = "data/tests/sentinel_results"
MODEL_PATH = "checkpoints/maps_test_2/latest_net_G_A.pth"
DEVICE = "cpu"
SIZE = 128
SAMPLE_SIZE = 20
SEED = 42

# Cargando modelo pre-entrenado
netG = networks.define_G(
    input_nc=3,
    output_nc=3,
    ngf=64,
    netG="resnet_9blocks",
    norm="instance",
    use_dropout=False,
    init_type="normal",
    init_gain=0.02,
)

netG.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
netG.to(DEVICE)
netG.eval()

preprocess = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229,0.224,0.225])
])

to_pil = transforms.ToPILImage()

# Como nuestras imágenes no son todas iguales en tamaño, esto las hace 128x128 para que funcionen bien con la función para calcular el FID
def resize_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for f in os.listdir(input_dir):
        path = os.path.join(input_dir, f)
        out_path = os.path.join(output_dir, f)

        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((SIZE, SIZE))
            img.save(out_path)
        except:
            continue

# Incluyendo nuestra propia métrica que creamos en clase, la llamamos YAGM (Yet Another GAN Metric)
def compute_yagm(fake_img, real_img):
    fake = np.array(fake_img).astype('float32')
    real = np.array(real_img).astype('float32')

    fake_mean = fake.mean()
    real_mean = real.mean()

    fake_diff = fake - fake_mean
    real_diff = real - real_mean

    fake_norm = linalg.norm(fake_diff)
    real_norm = linalg.norm(real_diff)

    return np.abs(real_norm - fake_norm) # Para que sea un valor entero siempre


def generate_images_compute_yagm():
    os.makedirs(FAKE_DIR, exist_ok=True)
    yagm_scores = [] # For our metric
    all_input_images = sorted(os.listdir(INPUT_DIR))
    all_real_styles = sorted(os.listdir(REAL_DIR))

    random.seed(SEED)
    
    input_sample = random.sample(all_input_images, min(SAMPLE_SIZE, len(all_input_images)))
    real_sample = random.sample(all_real_styles, min(SAMPLE_SIZE, len(all_real_styles)))

    for i, f in enumerate(input_sample):
        path = os.path.join(INPUT_DIR, f)
        real_style_path = os.path.join(REAL_RESIZED, real_sample[i])

        try:
            img_input = Image.open(path).convert("RGB")
            img_style = Image.open(real_style_path).convert("RGB")
        except:
            continue

        x = preprocess(img_input).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            y = netG(x)

        y = (y.squeeze().cpu() + 1) / 2
        y = to_pil(y).resize((SIZE, SIZE))

        y.save(os.path.join(FAKE_DIR, f))

        # Sacando YAGM
        img_real = img_style.resize((SIZE, SIZE))
        yagm = compute_yagm(y, img_real)
        yagm_scores.append(yagm)
    
    return yagm_scores


def compute_fid():
    fid = fid_score.calculate_fid_given_paths(
        [REAL_RESIZED, FAKE_DIR],
        batch_size=16,
        device=DEVICE,
        dims=2048,
        num_workers=0   
    )
    return fid

if __name__ == "__main__":
    # Estas funciones toman las imágenes de entrenamiento, las hace 
    resize_folder(REAL_DIR, REAL_RESIZED)
    yagm_scores = generate_images_compute_yagm()
    fid_value = compute_fid()
    
    print(f"\nFID score: {fid_value:.2f}")
    print(f"YAGM Score: {np.mean(yagm_scores):.2f}")
    print(f"YAGM STD: {np.std(yagm_scores):.2f}")