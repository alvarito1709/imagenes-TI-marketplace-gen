"""Análisis visual con RT-DETR (Hugging Face): ¿se conservó el contenido?

Idea: detectar objetos en la foto original y en la procesada. Si aparece o
desaparece algo, hay que avisarle a la persona usuaria.

Limitación importante: RT-DETR se entrenó con las 80 clases de COCO (personas,
bolsos, mochilas, corbatas, valijas...). NO tiene remeras, camperas ni zapatillas,
así que en muchas fotos de ropa no detecta nada. Cuando eso pasa, el informe lo
dice en vez de dar la foto por buena.
"""
from collections import Counter

from PIL import Image

MODELO_DETECTOR = "PekingU/rtdetr_r50vd"
UMBRAL_CONFIANZA = 0.7

# Nombres en castellano para los objetos que más probablemente aparezcan en fotos
# de productos. Los que no estén acá se muestran con el nombre original (inglés).
NOMBRES = {
    "person": "persona", "handbag": "bolso", "backpack": "mochila", "tie": "corbata",
    "suitcase": "valija", "umbrella": "paraguas", "bottle": "botella", "cup": "taza",
    "chair": "silla", "couch": "sillón", "bed": "cama", "dining table": "mesa",
    "laptop": "notebook", "cell phone": "celular", "tv": "televisor", "book": "libro",
    "clock": "reloj", "vase": "florero", "teddy bear": "peluche", "potted plant": "planta",
    "bicycle": "bicicleta", "sports ball": "pelota", "skateboard": "skate",
    "remote": "control remoto", "keyboard": "teclado", "mouse": "mouse",
}

_detector = None  # caché: el modelo se carga una sola vez


def cargar_detector():
    """Carga el detector UNA sola vez (pipeline 'object-detection')."""
    global _detector
    if _detector is None:
        import torch
        from transformers import pipeline  # import acá: no se paga al importar el módulo

        _detector = pipeline(
            "object-detection",
            model=MODELO_DETECTOR,
            device=0 if torch.cuda.is_available() else -1,
        )
    return _detector


def detectar_objetos(imagen: Image.Image, umbral: float = UMBRAL_CONFIANZA) -> list[dict]:
    """Lista de detecciones {'label', 'score', 'box'} con score >= umbral."""
    detecciones = cargar_detector()(imagen.convert("RGB"))
    return [d for d in detecciones if d["score"] >= umbral]


def _nombres(detecciones: list[dict]) -> Counter:
    return Counter(NOMBRES.get(d["label"], d["label"]) for d in detecciones)


def comparar_objetos(antes: list[dict], despues: list[dict]) -> dict:
    """Compara detecciones respetando cantidades (2 personas → 3 personas = 1 apareció).

    Devuelve {'aparecieron': [...], 'desaparecieron': [...], 'sin_cambios': [...]}
    con nombres de objetos en castellano.
    """
    a, d = _nombres(antes), _nombres(despues)
    return {
        "aparecieron": sorted((d - a).elements()),
        "desaparecieron": sorted((a - d).elements()),
        "sin_cambios": sorted((a & d).elements()),
    }
