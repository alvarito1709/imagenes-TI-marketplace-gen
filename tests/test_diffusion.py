"""Tests de diffusion.py que NO cargan el modelo (corren en cualquier PC)."""
import pytest
from PIL import Image

from src.diffusion import INTENSIDAD_MINIMA, _multiplo_de_8, restaurar


def test_multiplo_de_8_redondea_hacia_abajo():
    assert _multiplo_de_8((515, 701)) == (512, 696)


def test_multiplo_de_8_deja_igual_lo_que_ya_es_multiplo():
    assert _multiplo_de_8((512, 768)) == (512, 768)


def test_multiplo_de_8_minimo_8():
    assert _multiplo_de_8((3, 5)) == (8, 8)


@pytest.mark.parametrize("valor", [-0.1, 1.5])
def test_intensidad_fuera_de_rango_da_error(valor):
    with pytest.raises(ValueError):
        restaurar(Image.new("RGB", (64, 64)), valor)


def test_intensidad_casi_cero_devuelve_la_misma_imagen_sin_cargar_modelo():
    original = Image.new("RGB", (64, 64), (10, 20, 30))
    resultado = restaurar(original, INTENSIDAD_MINIMA / 2)
    assert resultado.size == original.size and resultado.getpixel((0, 0)) == (10, 20, 30)
