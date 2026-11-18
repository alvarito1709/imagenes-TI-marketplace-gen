import numpy as np
from PIL import Image

from src.metricas import degradar, psnr, ssim, tabla_markdown


def _foto(tam=(64, 64)):
    x = np.linspace(0, 255, tam[0], dtype=np.float32)
    y = np.linspace(0, 255, tam[1], dtype=np.float32)
    base = (x[None, :] + y[:, None]) / 2
    rojo = np.clip(base + 20 * np.sin(base / 8), 0, 255)
    return Image.fromarray(np.stack([rojo, base, 255 - base], axis=2).astype(np.uint8))


def test_psnr_es_infinito_si_son_identicas():
    assert psnr(_foto(), _foto()) == float("inf")


def test_psnr_baja_cuando_hay_mas_ruido():
    foto = _foto()
    assert psnr(foto, degradar(foto, 0, 5)) > psnr(foto, degradar(foto, 0, 40))


def test_ssim_es_uno_si_son_identicas_y_baja_con_ruido():
    foto = _foto()
    assert abs(ssim(foto, foto) - 1.0) < 1e-6
    assert ssim(foto, degradar(foto, 0, 30)) < 0.95


def test_ssim_penaliza_mas_el_desenfoque_fuerte():
    foto = _foto((96, 96))
    ruidosa = np.asarray(foto).copy()
    ruidosa[::4, ::4] = 255          # textura fina que el desenfoque borra
    ruidosa = Image.fromarray(ruidosa)
    assert ssim(ruidosa, degradar(ruidosa, 0.5, 0)) > ssim(ruidosa, degradar(ruidosa, 3.0, 0))


def test_metricas_aceptan_tamanos_distintos():
    assert psnr(_foto((64, 64)), _foto((32, 32))) > 0
    assert 0 < ssim(_foto((64, 64)), _foto((32, 32))) <= 1


def test_degradar_es_reproducible_con_la_misma_semilla_y_distinta_con_otra():
    foto = _foto()
    assert np.array_equal(np.asarray(degradar(foto, 1, 10, semilla=1)), np.asarray(degradar(foto, 1, 10, semilla=1)))
    assert not np.array_equal(np.asarray(degradar(foto, 1, 10, semilla=1)), np.asarray(degradar(foto, 1, 10, semilla=2)))


def test_degradar_mantiene_tamano_y_modo_y_sin_parametros_no_cambia_nada():
    foto = _foto()
    out = degradar(foto, 0, 0)
    assert out.size == foto.size and out.mode == "RGB"
    assert np.array_equal(np.asarray(out), np.asarray(foto))


def test_tabla_markdown_tiene_encabezado_y_una_fila_por_metodo():
    t = tabla_markdown([("A", 30.123, 0.9), ("B", 25.0, 0.5)])
    assert t.count("\n") == 3 and "30.12" in t and "0.500" in t
