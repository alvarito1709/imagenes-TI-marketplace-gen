from PIL import Image

from src.imagenes import LADO_MAXIMO, a_rgb, achicar, preparar


def test_a_rgb_convierte_png_con_transparencia():
    assert a_rgb(Image.new("RGBA", (10, 10))).mode == "RGB"


def test_achicar_respeta_lado_maximo_y_proporcion():
    resultado = achicar(Image.new("RGB", (2000, 1000)), 768)
    assert max(resultado.size) == 768
    assert resultado.size == (768, 384)


def test_achicar_no_agranda_imagenes_chicas():
    assert achicar(Image.new("RGB", (100, 50))).size == (100, 50)


def test_preparar_devuelve_rgb_acotado():
    resultado = preparar(Image.new("RGBA", (3000, 3000)))
    assert resultado.mode == "RGB" and max(resultado.size) <= LADO_MAXIMO
