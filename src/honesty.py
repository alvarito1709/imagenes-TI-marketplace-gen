"""Honestidad sobre lo generado: qué es real y qué es inventado.

Es el núcleo del trabajo: la IA no "recupera" detalle, lo reconstruye.

ESTADO: esqueleto.
"""
from PIL import Image

AVISO_IA = (
    "Esta imagen fue reconstruida por una IA. Los detalles finos no estaban en tu "
    "foto: revisalos antes de publicarla, sobre todo manchas, desgaste o defectos "
    "del producto."
)


def mapa_de_diferencias(original: Image.Image, resultado: Image.Image) -> Image.Image:
    """Mapa de calor de dónde cambió más la imagen respecto del original."""
    raise NotImplementedError("Pendiente")


def generar_informe(comparacion_objetos: dict, intensidad: float) -> str:
    """Informe en lenguaje simple (Markdown) para la persona usuaria."""
    raise NotImplementedError("Pendiente")
