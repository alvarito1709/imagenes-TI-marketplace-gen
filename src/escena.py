"""Escena nueva: la IA genera SOLO el fondo y el producto se pega sin tocarlo.

Pasos:
  1. Stable Diffusion dibuja un fondo vacío, SIN ver el producto (si lo ve, "completa" lo que
     cree que falta: una persona que lleva el bolso, un soporte de estudio, pelo...).
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
LADO_GENERADO = 512  # SD1.5 se entrenó con cuadrados de 512 px

# SD1.5 no entiende negaciones dentro del prompt ("no objects" las atrae): lo que no
# queremos va en el prompt negativo, y el positivo solo describe una superficie.
_BASE_NEGATIVO = ("person, people, human, face, body, hands, clothing, bag, shoes, jewelry, furniture, "
                  "objects, stand, tripod, curtain, backdrop, frame, border, text, logo, watermark, "
                  "animal, fur, shadow, pattern, clutter, blur, lowres")

# Los fondos de IA se definen por texto. "Blanco liso" no usa IA: ver `fondo_liso`.
# No usar palabras como "studio" o "backdrop": el modelo las asocia con soportes y telones.
FONDOS = {
    "Blanco de estudio": "plain white wall, smooth matte surface, soft even lighting, minimal, high key",
    "Madera clara": "close-up top view of a light oak wood table surface, natural grain, "
                    "soft even lighting, flat",
    "Gris suave": "plain light gray wall, smooth matte surface, soft even lighting, minimal",
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


def _cubrir(imagen: Image.Image, tamano: tuple[int, int]) -> Image.Image:
    """Escala `imagen` hasta cubrir `tamano` y recorta el centro, sin deformar."""
    ancho, alto = tamano
    escala = max(ancho / imagen.width, alto / imagen.height)
    nueva = imagen.resize((max(ancho, round(imagen.width * escala)), max(alto, round(imagen.height * escala))),
                          Image.LANCZOS)
    izq, arriba = (nueva.width - ancho) // 2, (nueva.height - alto) // 2
    return nueva.crop((izq, arriba, izq + ancho, arriba + alto))


def generar_fondo(tamano: tuple[int, int], fondo: str) -> Image.Image:
    """Genera un fondo vacío (solo la superficie descripta) del `tamano` pedido.

    El modelo NO recibe la foto del producto: se pide una imagen entera nueva en un cuadrado
    (el formato con el que se entrenó SD1.5; en formatos altos duplica objetos) y después se
    ajusta al formato de la foto. Se usa el pipeline de inpainting con la máscara completa,
    así no hay que descargar otro modelo.
    """
    if fondo not in FONDOS:
        raise ValueError(f"Fondo desconocido: {fondo!r}. Opciones con IA: {list(FONDOS)}")
    import torch

    pipe = cargar_inpainting()
    generador = torch.Generator(device=pipe.device).manual_seed(SEMILLA)
    salida = pipe(
        prompt=FONDOS[fondo],
        negative_prompt=_BASE_NEGATIVO,
        image=Image.new("RGB", (LADO_GENERADO, LADO_GENERADO), (128, 128, 128)),
        mask_image=Image.new("L", (LADO_GENERADO, LADO_GENERADO), 255),
        width=LADO_GENERADO, height=LADO_GENERADO,
        num_inference_steps=PASOS,
        guidance_scale=GUIA,
        generator=generador,
    ).images[0]
    return _cubrir(salida, tamano)


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
