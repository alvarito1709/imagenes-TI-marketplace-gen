"""Procesamiento con difusión: Stable Diffusion 1.5 + ControlNet tile (img2img).

ESTADO: esqueleto. La lógica se implementa en la rama feature/modelado.
"""
from PIL import Image

MODELO_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODELO_CONTROLNET = "lllyasviel/control_v11f1e_sd15_tile"


def cargar_pipeline():
    """Carga el pipeline UNA sola vez (al iniciar la app), en float16 si hay GPU.

    Devuelve el pipeline de diffusers listo para usar.
    """
    raise NotImplementedError("Pendiente: feature/modelado")


def restaurar(imagen: Image.Image, intensidad: float = 0.3) -> Image.Image:
    """Limpia ruido/desenfoque de `imagen`.

    `intensidad` (0.0-1.0) es el `strength` del img2img: cuánto puede cambiar el
    modelo la foto. Valores bajos conservan más del original (más honesto).
    """
    raise NotImplementedError("Pendiente: feature/modelado")
