"""Carga y preparación básica de imágenes (Pillow)."""
from PIL import Image

LADO_MAXIMO = 768  # px; mantiene el uso de memoria acotado en la GPU de Colab


def a_rgb(imagen: Image.Image) -> Image.Image:
    """Convierte a RGB. Necesario para PNG con transparencia, escala de grises, etc."""
    return imagen.convert("RGB")


def achicar(imagen: Image.Image, lado_maximo: int = LADO_MAXIMO) -> Image.Image:
    """Devuelve una copia cuyo lado mayor no supera `lado_maximo`, sin deformar."""
    copia = imagen.copy()
    copia.thumbnail((lado_maximo, lado_maximo))
    return copia


def preparar(imagen: Image.Image, lado_maximo: int = LADO_MAXIMO) -> Image.Image:
    """Pipeline de entrada: RGB + tamaño máximo."""
    return achicar(a_rgb(imagen), lado_maximo)
