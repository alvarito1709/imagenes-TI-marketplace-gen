import numpy as np
import pytest
from PIL import Image

from src.escena import FONDO_SIN_IA, FONDOS, OPCIONES_FONDO, componer, fondo_liso, generar_fondo


def _aleatoria(tam=(32, 32), semilla=0):
    rng = np.random.default_rng(semilla)
    return Image.fromarray(rng.integers(0, 256, (tam[1], tam[0], 3), dtype=np.uint8))


def test_donde_alfa_es_uno_el_producto_queda_identico():
    producto = _aleatoria()
    alfa = np.ones((32, 32), np.float32)
    assert np.array_equal(np.asarray(componer(producto, fondo_liso((32, 32)), alfa)), np.asarray(producto))


def test_donde_alfa_es_cero_se_ve_el_fondo():
    fondo = _aleatoria(semilla=1)
    alfa = np.zeros((32, 32), np.float32)
    assert np.array_equal(np.asarray(componer(_aleatoria(), fondo, alfa)), np.asarray(fondo))


def test_el_producto_no_cambia_aunque_el_fondo_sea_distinto():
    producto = _aleatoria()
    alfa = np.zeros((32, 32), np.float32)
    alfa[8:24, 8:24] = 1.0
    a = np.asarray(componer(producto, fondo_liso((32, 32), (255, 255, 255)), alfa))
    b = np.asarray(componer(producto, fondo_liso((32, 32), (10, 10, 10)), alfa))
    assert np.array_equal(a[8:24, 8:24], np.asarray(producto)[8:24, 8:24])
    assert np.array_equal(a[8:24, 8:24], b[8:24, 8:24])
    assert not np.array_equal(a[:4], b[:4])


def test_alfa_intermedio_mezcla():
    producto = Image.new("RGB", (4, 4), (200, 200, 200))
    fondo = Image.new("RGB", (4, 4), (0, 0, 0))
    out = componer(producto, fondo, np.full((4, 4), 0.5, np.float32))
    assert out.getpixel((1, 1)) == (100, 100, 100)


def test_componer_acepta_fondo_de_otro_tamano():
    assert componer(_aleatoria((32, 32)), fondo_liso((64, 64)), np.ones((32, 32), np.float32)).size == (32, 32)


def test_fondo_liso_es_del_color_pedido():
    assert fondo_liso((5, 5), (1, 2, 3)).getpixel((2, 2)) == (1, 2, 3)


def test_opciones_de_fondo_incluyen_la_comparacion_sin_ia():
    assert OPCIONES_FONDO == [*FONDOS, FONDO_SIN_IA] and FONDO_SIN_IA not in FONDOS
    assert len(FONDOS) == 3


def test_generar_fondo_rechaza_un_fondo_desconocido_sin_cargar_el_modelo():
    with pytest.raises(ValueError):
        generar_fondo(_aleatoria(), Image.new("L", (32, 32)), "Playa")
