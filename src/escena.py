"""Escena nueva: la IA genera SOLO el fondo y el producto se pega sin tocarlo.

Pasos:
  1. Inpainting (Stable Diffusion) dibuja un fondo nuevo en la zona que NO es producto.
  2. `componer` pega encima los píxeles originales del producto.

Resultado: logos, botones, estampados y costuras son los de la foto real.
Lo único inventado es el fondo.
"""
import numpy as np
from PIL import Image

MODELO_INPAINTING = "stable-diffusion-v1-5/stable-diffusion-inpainting"  # CreativeML OpenRAIL-M

PASOS = 30
SEMILLA = 0
GUIA = 7.5  # cuánto obedece el modelo al texto

_BASE_NEGATIVO = ("product, clothing, bag, shoes, jewelry, person, hands, text, logo, "
                  "watermark, shadow of an object, clutter, blur, lowres")

# Los fondos de IA se definen por texto. "Blanco liso" no usa IA: ver `fondo_liso`.
FONDOS = {
    "Blanco de estudio": "empty seamless white photo studio backdrop, soft diffused light, "
                         "clean product photography background, no objects",
    "Madera clara": "empty light oak wooden table surface, soft natural window light, "
                    "clean product photography background, no objects",
    "Gris suave": "empty soft light gray seamless paper backdrop, even soft lighting, "
                  "clean product photography background, no objects",
}
FONDO_SIN_IA = "Blanco liso (sin IA)"
OPCIONES_FONDO = [*FONDOS, FONDO_SIN_IA]

_pipeline = None  # caché: el modelo se carga una sola vez


def _dispositivo_y_tipo():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def cargar_inpainting():
    """Carga el pipeline de inpainting UNA sola vez (import perezoso)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    from diffusers import StableDiffusionInpaintPipeline

    dispositivo, tipo = _dispositivo_y_tipo()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODELO_INPAINTING,
        torch_dtype=tipo,
        # Mismo criterio que en diffusion.py (decisión registrada en docs/decisiones.md):
        # sobre fotos de ropa el filtro da falsos positivos y devuelve una imagen negra.
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.to(dispositivo)
    _pipeline = pipe
    return _pipeline


def _multiplo_de_8(valor: int) -> int:
    return max(8, valor // 8 * 8)


def generar_fondo(imagen: Image.Image, mascara_fondo: Image.Image, fondo: str) -> Image.Image:
    """Devuelve la foto con la zona blanca de `mascara_fondo` regenerada según `fondo`.

    El tamaño de salida es el de la foto de entrada. Solo hay que usar el fondo de este
    resultado: el producto que el modelo "redibuja" se descarta al componer.
    """
    if fondo not in FONDOS:
        raise ValueError(f"Fondo desconocido: {fondo!r}. Opciones con IA: {list(FONDOS)}")
    import torch

    pipe = cargar_inpainting()
    ancho, alto = imagen.size
    medidas = (_multiplo_de_8(ancho), _multiplo_de_8(alto))
    generador = torch.Generator(device=pipe.device).manual_seed(SEMILLA)
    salida = pipe(
        prompt=FONDOS[fondo],
        negative_prompt=_BASE_NEGATIVO,
        image=imagen.convert("RGB").resize(medidas, Image.LANCZOS),
        mask_image=mascara_fondo.convert("L").resize(medidas, Image.NEAREST),
        width=medidas[0], height=medidas[1],
        num_inference_steps=PASOS,
        guidance_scale=GUIA,
        generator=generador,
    ).images[0]
    return salida.resize((ancho, alto), Image.LANCZOS)


def fondo_liso(tamano: tuple[int, int], color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Fondo de un solo color, sin IA. Sirve de comparación honesta: ¿la IA aporta algo?"""
    return Image.new("RGB", tamano, color)


def componer(producto: Image.Image, fondo: Image.Image, alfa: np.ndarray) -> Image.Image:
    """Pega `producto` sobre `fondo` con la transparencia `alfa` (float 0-1, alto x ancho).

    Donde alfa = 1 el píxel es EXACTAMENTE el de `producto`; el modelo no lo toca.
    """
    if fondo.size != producto.size:
        fondo = fondo.resize(producto.size, Image.LANCZOS)
    p = np.asarray(producto.convert("RGB"), dtype=np.float32)
    f = np.asarray(fondo.convert("RGB"), dtype=np.float32)
    a = np.clip(alfa, 0.0, 1.0)[..., None]
    mezcla = p * a + f * (1.0 - a)
    return Image.fromarray(np.rint(mezcla).astype(np.uint8), mode="RGB")
