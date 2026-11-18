"""Procesamiento con difusión: Stable Diffusion 1.5 + ControlNet tile (img2img).

Idea: el modelo recibe la foto y la "vuelve a dibujar" parcialmente. ControlNet
tile lo obliga a respetar la estructura de la foto original (formas, bordes,
colores). El parámetro `intensidad` (strength) controla cuánto se aleja del
original: bajo = conservador, alto = el modelo inventa más.

Todo lo que el modelo agrega NO estaba en la foto: es una reconstrucción.
"""
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image

MODELO_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODELO_CONTROLNET = "lllyasviel/control_v11f1e_sd15_tile"

# Prompt neutro a propósito: no describe el producto, para no inducir al modelo a
# inventar contenido. Solo pide calidad.
PROMPT = "best quality, sharp, natural lighting"
PROMPT_NEGATIVO = "blur, lowres, noise, jpeg artifacts, worst quality"

PASOS = 30        # pasos de difusión; con strength s se ejecutan ~PASOS * s
SEMILLA = 0       # fija, para que el mismo input dé el mismo resultado
INTENSIDAD_MINIMA = 0.05

_pipeline = None  # caché: el modelo se carga una sola vez


def _dispositivo_y_tipo() -> tuple[str, torch.dtype]:
    """GPU + float16 si hay CUDA (Colab); si no, CPU + float32 (muy lento)."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def cargar_pipeline():
    """Carga el pipeline UNA sola vez y lo reutiliza en las llamadas siguientes."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    dispositivo, tipo = _dispositivo_y_tipo()
    controlnet = ControlNetModel.from_pretrained(MODELO_CONTROLNET, torch_dtype=tipo)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        MODELO_BASE,
        controlnet=controlnet,
        torch_dtype=tipo,
        # Sin filtro de seguridad: sobre fotos de ropa suele dar falsos positivos
        # y devuelve una imagen negra. Decisión registrada en docs/decisiones.md.
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(dispositivo)
    _pipeline = pipe
    return _pipeline


def _multiplo_de_8(tamano: tuple[int, int]) -> tuple[int, int]:
    """SD trabaja con lados múltiplos de 8; redondea hacia abajo (mínimo 8)."""
    ancho, alto = tamano
    return max(8, ancho - ancho % 8), max(8, alto - alto % 8)


def restaurar(imagen: Image.Image, intensidad: float = 0.3) -> Image.Image:
    """Limpia ruido/desenfoque de `imagen` y devuelve una imagen del MISMO tamaño.

    `intensidad` (0.0-1.0) es el `strength` del img2img: cuánto puede cambiar el
    modelo la foto. Valores bajos conservan más del original (más honesto).
    """
    if not 0.0 <= intensidad <= 1.0:
        raise ValueError("La intensidad tiene que estar entre 0 y 1.")
    if intensidad < INTENSIDAD_MINIMA:
        return imagen.copy()  # sin cambios: no hay nada que reconstruir

    pipe = cargar_pipeline()
    tamano_original = imagen.size
    entrada = imagen.convert("RGB").resize(_multiplo_de_8(tamano_original), Image.LANCZOS)

    dispositivo, _ = _dispositivo_y_tipo()
    generador = torch.Generator(device=dispositivo).manual_seed(SEMILLA)
    resultado = pipe(
        prompt=PROMPT,
        negative_prompt=PROMPT_NEGATIVO,
        image=entrada,          # imagen que se re-dibuja parcialmente
        control_image=entrada,  # ControlNet tile: ancla la estructura al original
        strength=intensidad,
        num_inference_steps=PASOS,
        generator=generador,
    ).images[0]

    # Volvemos al tamaño original para poder comparar píxel a píxel (mapa de diferencias).
    return resultado.resize(tamano_original, Image.LANCZOS)
