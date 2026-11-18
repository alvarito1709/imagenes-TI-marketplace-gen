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


# --- enfoque "producto real + fondo nuevo" ---
import numpy as np  # noqa: E402

from src.honesty import AVISO_ESCENA, generar_informe_escena, generar_informe_fondo, porcentaje_producto_modificado  # noqa: E402


def _producto(tam=40):
    m = np.zeros((tam, tam), bool)
    m[8:32, 8:32] = True
    return m


def test_producto_modificado_es_cero_si_el_producto_no_cambia():
    original = _gris(100, (40, 40))
    resultado = original.copy()
    resultado.paste(_gris(240, (40, 5)), (0, 0))        # cambia solo el fondo (franja)
    assert porcentaje_producto_modificado(original, resultado, _producto()) == 0.0


def test_producto_modificado_detecta_cambios_en_el_producto():
    original = _gris(100, (40, 40))
    resultado = original.copy()
    resultado.paste(_gris(240, (24, 24)), (8, 8))
    assert porcentaje_producto_modificado(original, resultado, _producto()) > 90


def test_producto_modificado_con_mascara_vacia_no_falla():
    assert porcentaje_producto_modificado(_gris(1), _gris(200), np.zeros((32, 32), bool)) == 0.0


def test_informe_escena_dice_que_el_producto_es_real():
    texto = generar_informe_escena("Madera clara", True, 0.0)
    assert "Madera clara" in texto and "0%" in texto and "generado por una IA" in texto


def test_informe_escena_sin_ia_lo_aclara():
    assert "No usa IA" in generar_informe_escena("Blanco liso (sin IA)", False, 0.0)


def test_informe_escena_atribuye_el_cambio_a_la_luz_si_estaba_activada():
    assert "iluminación" in generar_informe_escena("Gris suave", True, 12.5, iluminacion_corregida=True)


def test_informe_escena_alerta_si_cambio_sin_corregir_luz():
    assert "🔴" in generar_informe_escena("Gris suave", True, 12.5)


def test_aviso_escena_aclara_que_solo_el_fondo_es_ia():
    assert "fondo" in AVISO_ESCENA


def test_informe_fondo_distingue_lo_que_cambio_en_este_paso_de_lo_que_cambio_antes():
    texto = generar_informe_fondo("Madera clara", True, 0.0, 4.0)
    assert "no se tocó" in texto and "4.0%" in texto and "paso 1" in texto and "sin ver tu producto" in texto


def test_informe_fondo_alerta_si_el_producto_cambio_en_este_paso():
    assert "🔴" in generar_informe_fondo("Gris suave", True, 3.0, 3.0)


def test_informe_fondo_sin_ia_lo_aclara():
    assert "No usa IA" in generar_informe_fondo("Blanco liso (sin IA)", False, 0.0, 0.0)
