from PIL import Image

from src.honesty import AVISO_IA, generar_informe, mapa_de_diferencias, porcentaje_cambiado


def _gris(valor, tam=(32, 32)):
    return Image.new("RGB", tam, (valor, valor, valor))


def test_porcentaje_cero_si_son_iguales():
    assert porcentaje_cambiado(_gris(100), _gris(100)) == 0.0


def test_porcentaje_cien_si_cambia_todo():
    assert porcentaje_cambiado(_gris(0), _gris(200)) == 100.0


def test_porcentaje_ignora_cambios_chicos():
    assert porcentaje_cambiado(_gris(100), _gris(110)) == 0.0


def test_mapa_tiene_el_tamano_original_y_distingue_zonas():
    original = _gris(50, (40, 40))
    resultado = original.copy()
    resultado.paste(_gris(250, (20, 40)), (0, 0))   # mitad izquierda cambia mucho
    mapa = mapa_de_diferencias(original, resultado)
    assert mapa.size == original.size and mapa.mode == "RGB"
    r_izq, _, b_izq = mapa.getpixel((3, 20))
    r_der, _, b_der = mapa.getpixel((36, 20))
    assert r_izq > r_der and b_der > b_izq   # izquierda "caliente", derecha "fría"


def test_mapa_acepta_resultado_de_otro_tamano():
    assert mapa_de_diferencias(_gris(10, (32, 32)), _gris(10, (64, 64))).size == (32, 32)


def test_informe_avisa_objetos_nuevos_y_perdidos():
    texto = generar_informe({"aparecieron": ["persona"], "desaparecieron": ["bolso"]}, 0.3, 4.2)
    assert "persona" in texto and "bolso" in texto and "4.2%" in texto


def test_informe_admite_que_no_pudo_verificar_si_no_detecta_nada():
    assert "no puede confirmar" in generar_informe({}, 0.3)


def test_informe_advierte_con_intensidad_alta():
    assert "inventado" in generar_informe({}, 0.8)


def test_aviso_menciona_ia():
    assert "IA" in AVISO_IA


def test_informe_menciona_la_correccion_de_luz_solo_si_se_aplico():
    assert "Iluminación" in generar_informe({}, 0.3, None, True)
    assert "Iluminación" not in generar_informe({}, 0.3, None, False)
