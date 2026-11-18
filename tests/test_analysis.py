"""Tests de analysis.py con un detector falso (no descargan el modelo)."""
from PIL import Image

from src import analysis
from src.analysis import comparar_objetos, detectar_objetos


def _det(label, score=0.9):
    return {"label": label, "score": score, "box": {}}


def test_sin_cambios():
    r = comparar_objetos([_det("person")], [_det("person")])
    assert r == {"aparecieron": [], "desaparecieron": [], "sin_cambios": ["persona"]}


def test_aparece_y_desaparece_con_nombres_en_castellano():
    r = comparar_objetos([_det("handbag")], [_det("person")])
    assert r["aparecieron"] == ["persona"] and r["desaparecieron"] == ["bolso"]


def test_respeta_cantidades():
    r = comparar_objetos([_det("person")] * 2, [_det("person")] * 3)
    assert r["aparecieron"] == ["persona"] and r["sin_cambios"] == ["persona", "persona"]


def test_label_desconocido_se_deja_como_viene():
    assert comparar_objetos([], [_det("zebra")])["aparecieron"] == ["zebra"]


def test_listas_vacias():
    assert comparar_objetos([], []) == {"aparecieron": [], "desaparecieron": [], "sin_cambios": []}


def test_detectar_filtra_por_umbral():
    anterior = analysis._detector
    analysis._detector = lambda imagen: [_det("person", 0.95), _det("tie", 0.40)]
    try:
        resultado = detectar_objetos(Image.new("RGB", (8, 8)), 0.7)
    finally:
        analysis._detector = anterior
    assert [d["label"] for d in resultado] == ["person"]
