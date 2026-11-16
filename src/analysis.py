"""Análisis visual con RT-DETR (Hugging Face): ¿se conservó el contenido?

Idea: detectar objetos en la foto original y en la procesada. Si aparece o
desaparece algo, hay que avisarle a la persona usuaria.

ESTADO: esqueleto. Modelo: PekingU/rtdetr_r50vd
"""
from PIL import Image

MODELO_DETECTOR = "PekingU/rtdetr_r50vd"
UMBRAL_CONFIANZA = 0.7


def cargar_detector():
    """Carga el detector UNA sola vez (pipeline 'object-detection')."""
    raise NotImplementedError("Pendiente")


def detectar_objetos(imagen: Image.Image, umbral: float = UMBRAL_CONFIANZA) -> list[dict]:
    """Lista de detecciones con score >= umbral."""
    raise NotImplementedError("Pendiente")


def comparar_objetos(antes: list[dict], despues: list[dict]) -> dict:
    """Compara detecciones. Devuelve p. ej. {'aparecieron': [...], 'desaparecieron': [...]}."""
    raise NotImplementedError("Pendiente")
