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


def corregir_iluminacion(imagen: Image.Image, limite_contraste: float = 2.0) -> Image.Image:
    """Mejora fotos oscuras o con luz pareja sin inventar contenido.

    Trabaja solo sobre la luminosidad (canal L del espacio L*a*b*), sin tocar el color:
    1. Gamma automático: si la foto es oscura, la aclara hasta una media razonable.
       Nunca oscurece una foto que ya es clara.
    2. CLAHE (ecualización adaptativa de histograma): reparte mejor el contraste por zonas.

    Limitación: no recupera lo que la cámara no registró y puede resaltar el ruido.
    """
    lab = cv2.cvtColor(np.asarray(imagen.convert("RGB")), cv2.COLOR_RGB2LAB)
    luz = lab[:, :, 0]

    media = float(np.clip(luz.mean() / 255.0, 0.02, 0.98))
    gamma = float(np.clip(np.log(0.5) / np.log(media), 0.5, 1.0))  # <1 aclara, 1 no cambia
    tabla = (255.0 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
    luz = cv2.LUT(luz, tabla)

    luz = cv2.createCLAHE(clipLimit=limite_contraste, tileGridSize=(8, 8)).apply(luz)
    lab[:, :, 0] = luz
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
