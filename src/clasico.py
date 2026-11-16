"""Línea base clásica (sin IA) para comparar contra la difusión.

ESTADO: esqueleto.
"""
from PIL import Image


def filtro_bilateral(imagen: Image.Image) -> Image.Image:
    """Reduce ruido preservando bordes (OpenCV)."""
    raise NotImplementedError("Pendiente")


def filtro_mediano(imagen: Image.Image, tamano: int = 3) -> Image.Image:
    """Reduce ruido tipo sal y pimienta (OpenCV)."""
    raise NotImplementedError("Pendiente")
