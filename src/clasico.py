"""Línea base clásica (sin IA) para comparar contra la difusión.

Sirve para ver qué agrega realmente el modelo: si el filtro clásico ya limpia
la foto, la difusión tiene que justificar el costo. Los filtros clásicos no
inventan detalle: solo suavizan lo que ya está en la imagen.
"""
import cv2
import numpy as np
from PIL import Image


def filtro_bilateral(imagen: Image.Image, diametro: int = 9,
                     sigma_color: float = 75, sigma_espacio: float = 75) -> Image.Image:
    """Reduce ruido suavizando zonas parejas pero preservando los bordes."""
    arreglo = np.asarray(imagen.convert("RGB"))
    filtrado = cv2.bilateralFilter(arreglo, diametro, sigma_color, sigma_espacio)
    return Image.fromarray(filtrado)


def filtro_mediano(imagen: Image.Image, tamano: int = 3) -> Image.Image:
    """Reduce ruido tipo sal y pimienta reemplazando cada píxel por la mediana de su vecindario."""
    if tamano < 3 or tamano % 2 == 0:
        raise ValueError("El tamaño del filtro mediano tiene que ser impar y al menos 3.")
    arreglo = np.asarray(imagen.convert("RGB"))
    return Image.fromarray(cv2.medianBlur(arreglo, tamano))
