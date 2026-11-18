import numpy as np
from PIL import Image

from src.segmentacion import (
    alfa_para_composicion, binarizar, limpiar_mascara, mascara_para_inpainting,
    quitar_islas, rellenar_huecos,
)


def _cuadrado(tam=64, lado=24):
    m = np.zeros((tam, tam), bool)
    ini = (tam - lado) // 2
    m[ini:ini + lado, ini:ini + lado] = True
    return m


def test_binarizar_usa_el_umbral():
    img = Image.fromarray(np.array([[10, 127], [128, 250]], np.uint8), mode="L")
    assert binarizar(img).tolist() == [[False, False], [True, True]]


def test_quitar_islas_borra_manchas_chicas_y_conserva_el_producto():
    m = _cuadrado()
    m[2, 2] = True                      # mancha suelta de 1 píxel
    limpia = quitar_islas(m)
    assert not limpia[2, 2] and limpia[32, 32]


def test_quitar_islas_conserva_un_producto_en_dos_partes():
    m = np.zeros((64, 64), bool)
    m[10:30, 10:30] = True
    m[40:58, 40:58] = True              # segunda parte, tamaño parecido
    limpia = quitar_islas(m)
    assert limpia[20, 20] and limpia[50, 50]


def test_quitar_islas_con_mascara_vacia_no_falla():
    assert not quitar_islas(np.zeros((16, 16), bool)).any()


def test_rellenar_huecos_tapa_agujeros_chicos_pero_no_el_fondo():
    m = _cuadrado(128, 64)
    m[60:63, 60:63] = False             # agujero chico adentro (un botón)
    lleno = rellenar_huecos(m)
    assert lleno[61, 61] and not lleno[2, 2]


def test_rellenar_huecos_deja_los_huecos_grandes_como_fondo():
    m = _cuadrado(128, 64)              # producto con un hueco grande (espacio entre correas)
    m[48:64, 48:64] = False             # 256 px > 0.5% de 4096
    assert not rellenar_huecos(m)[55, 55]


def test_limpiar_mascara_junta_los_pasos():
    m = _cuadrado(128, 64)
    m[2, 2] = True
    m[60:62, 60:62] = False
    img = Image.fromarray(m.astype(np.uint8) * 255, mode="L")
    limpia = limpiar_mascara(img)
    assert not limpia[2, 2] and limpia[61, 61]


def test_alfa_es_uno_adentro_cero_afuera_y_gradual_en_el_borde():
    alfa = alfa_para_composicion(_cuadrado())
    assert alfa.dtype == np.float32 and 0.0 <= alfa.min() and alfa.max() <= 1.0
    assert alfa[32, 32] == 1.0 and alfa[2, 2] == 0.0
    borde = alfa[32, 20:26]             # cruza el borde izquierdo del cuadrado
    assert ((borde > 0) & (borde < 1)).any()


def test_alfa_se_achica_respecto_de_la_mascara():
    m = _cuadrado()
    assert (alfa_para_composicion(m) >= 0.99).sum() < m.sum()


def test_los_bordes_crecen_con_el_tamano_de_la_foto():
    chica = alfa_para_composicion(_cuadrado(768, 400))
    grande = alfa_para_composicion(_cuadrado(1536, 800))
    # en la foto del doble de lado, el borde se erosiona/difumina el doble: la zona intermedia es más ancha
    assert ((grande > 0) & (grande < 1)).sum(axis=1)[768] > ((chica > 0) & (chica < 1)).sum(axis=1)[384]


def test_mascara_de_inpainting_protege_el_producto_y_entra_en_su_borde():
    m = _cuadrado()
    mi = np.asarray(mascara_para_inpainting(m, expansion_px=4))
    assert mi[32, 32] == 0              # centro del producto: protegido
    assert mi[2, 2] == 255              # fondo: se genera
    assert mi[20, 32] == 255            # borde del producto (ini=20): ya cae en zona a generar
