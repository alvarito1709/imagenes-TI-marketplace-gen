import numpy as np
import pytest
from PIL import Image

from src.clasico import filtro_bilateral, filtro_mediano


def _ruidosa():
    ruido = np.random.default_rng(0).normal(128, 30, (64, 64, 3))
    return Image.fromarray(np.clip(ruido, 0, 255).astype("uint8"))


def test_bilateral_mantiene_tamano_y_reduce_ruido():
    original = _ruidosa()
    resultado = filtro_bilateral(original)
    assert resultado.size == original.size
    assert np.asarray(resultado).std() < np.asarray(original).std()


def test_mediano_mantiene_tamano_y_reduce_ruido():
    original = _ruidosa()
    resultado = filtro_mediano(original, 5)
    assert resultado.size == original.size
    assert np.asarray(resultado).std() < np.asarray(original).std()


@pytest.mark.parametrize("tamano", [2, 4, 1])
def test_mediano_rechaza_tamanos_invalidos(tamano):
    with pytest.raises(ValueError):
        filtro_mediano(Image.new("RGB", (8, 8)), tamano)


def test_acepta_png_con_transparencia():
    assert filtro_bilateral(Image.new("RGBA", (16, 16))).mode == "RGB"
