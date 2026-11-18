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


def _oscura():
    ruido = np.random.default_rng(1).normal(40, 12, (64, 64, 3))
    return Image.fromarray(np.clip(ruido, 0, 255).astype("uint8"))


def test_iluminacion_aclara_una_foto_oscura():
    from src.clasico import corregir_iluminacion
    original = _oscura()
    resultado = corregir_iluminacion(original)
    assert resultado.size == original.size
    assert np.asarray(resultado).mean() > np.asarray(original).mean() + 20


def test_iluminacion_no_oscurece_una_foto_clara():
    from src.clasico import corregir_iluminacion
    # Degradé claro con un poco de ruido: se parece a una foto bien expuesta
    base = np.tile(np.linspace(150, 240, 64), (64, 1))
    ruido = np.random.default_rng(2).normal(0, 4, (64, 64, 3))
    clara = Image.fromarray(np.clip(np.stack([base, base * .95, base * .9], axis=2) + ruido, 0, 255).astype("uint8"))
    assert abs(np.asarray(corregir_iluminacion(clara)).mean() - np.asarray(clara).mean()) < 10


def test_iluminacion_acepta_png_con_transparencia_y_negro_total():
    from src.clasico import corregir_iluminacion
    assert corregir_iluminacion(Image.new("RGBA", (16, 16))).mode == "RGB"
