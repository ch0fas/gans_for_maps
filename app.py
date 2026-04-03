import torch
import numpy as np
from PIL import Image
import gradio as gr

from models import networks

# Cargando Modelo pre-entrenado
device = "cpu"

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

netG.load_state_dict(
    torch.load("checkpoints/maps_test/latest_net_G_A.pth",
               map_location=device,
               weights_only=True)
)

# Procesamiento de Imagen para modelo
def preprocess(img):
    img = img.convert("RGB").resize((128, 128))
    img = np.array(img).astype("float32") / 127.5 - 1.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    return img

def postprocess(tensor):
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img + 1) * 127.5
    return img.astype("uint8")

# Generando nueva imagen
def translate(image):
    x = preprocess(image)
    with torch.no_grad():
        y = netG(x)
    return Image.fromarray(postprocess(y))

# Interfaz de Gradio
demo = gr.Interface(
    fn=translate,
    inputs=gr.Image(type="pil", height=800, width=800),
    outputs=gr.Image(type="pil", height=800, width=800),
    title="CycleGAN - OAM Version",
    description="Sube una imagen para convertirla en un mapa de ciudad medieval!",
)

demo.launch()