"""Honestidad sobre lo generado: qué es real y qué es inventado.

Es el núcleo del trabajo: la IA no "recupera" detalle, lo reconstruye.
Este módulo le muestra a la persona usuaria DÓNDE cambió la foto y cuánto.
"""
import cv2
import numpy as np
from PIL import Image

AVISO_IA = (
    "⚠️ **Esta imagen fue reconstruida por una IA.** Los detalles finos no estaban en tu "
    "foto: revisalos antes de publicarla, sobre todo manchas, desgaste o defectos "
    "del producto."
)

DIFERENCIA_VISIBLE = 25  # sobre 255: por debajo, el cambio no se nota a simple vista
ESCALA_MAPA = 64         # diferencia que se pinta como "rojo": fija, así "no cambió nada" se ve frío


def _diferencia(original: Image.Image, resultado: Image.Image) -> np.ndarray:
    """Diferencia absoluta por píxel (0-255), tomando el mayor cambio entre canales RGB."""
    if resultado.size != original.size:
        resultado = resultado.resize(original.size, Image.LANCZOS)
    a = np.asarray(original.convert("RGB"), dtype=np.int16)
    b = np.asarray(resultado.convert("RGB"), dtype=np.int16)
    return np.abs(a - b).max(axis=2).astype(np.uint8)


def mapa_de_diferencias(original: Image.Image, resultado: Image.Image) -> Image.Image:
    """Mapa de calor: azul = casi igual al original, rojo = cambió mucho."""
    dif = _diferencia(original, resultado)
    dif = cv2.GaussianBlur(dif, (0, 0), 2)  # suaviza para que se lean zonas, no píxeles sueltos
    escala = np.clip(dif.astype(np.float32) / ESCALA_MAPA * 255, 0, 255).astype(np.uint8)
    calor_bgr = cv2.applyColorMap(escala, cv2.COLORMAP_JET)
    return Image.fromarray(cv2.cvtColor(calor_bgr, cv2.COLOR_BGR2RGB))


def porcentaje_cambiado(original: Image.Image, resultado: Image.Image) -> float:
    """Porcentaje (0-100) de píxeles que cambiaron de forma visible respecto del original."""
    dif = _diferencia(original, resultado)
    return float((dif > DIFERENCIA_VISIBLE).mean() * 100)


def _nivel(intensidad: float) -> str:
    if intensidad <= 0.3:
        return "suave: el modelo cambió poco y conservó casi todo lo que había"
    if intensidad <= 0.6:
        return "media: el modelo pudo reconstruir texturas y detalles chicos"
    return "alta: el modelo tuvo mucha libertad y puede haber inventado detalles, incluso tapado defectos"


def generar_informe(comparacion_objetos: dict, intensidad: float,
                    porcentaje: float | None = None) -> str:
    """Informe en lenguaje simple (Markdown) para la persona usuaria.

    `comparacion_objetos` viene de analysis.comparar_objetos, con las claves
    'aparecieron', 'desaparecieron' y 'sin_cambios' (listas de nombres de objetos).
    """
    lineas = [f"**Intensidad usada:** {intensidad:.2f} → {_nivel(intensidad)}."]

    if porcentaje is not None:
        lineas.append(f"**Cuánto cambió la foto:** {porcentaje:.1f}% de la imagen se modificó de forma visible. "
                      "Mirá el mapa de colores: las zonas rojas son las que más cambiaron.")

    aparecieron = comparacion_objetos.get("aparecieron", [])
    desaparecieron = comparacion_objetos.get("desaparecieron", [])
    sin_cambios = comparacion_objetos.get("sin_cambios", [])

    if aparecieron or desaparecieron:
        if aparecieron:
            lineas.append(f"🔴 **Aparecieron objetos que no estaban:** {', '.join(aparecieron)}. "
                          "El modelo los inventó: revisá la foto.")
        if desaparecieron:
            lineas.append(f"🔴 **Desaparecieron objetos que sí estaban:** {', '.join(desaparecieron)}. "
                          "Verificá que no se haya borrado algo importante.")
    elif sin_cambios:
        lineas.append(f"🟢 Se detectaron los mismos objetos antes y después ({', '.join(sin_cambios)}).")
    else:
        lineas.append("ℹ️ El detector no encontró objetos conocidos (personas, bolsos, mochilas, etc.), "
                      "así que **no puede confirmar que el contenido se conservó**. "
                      "Compará vos misma la foto original con el resultado.")

    lineas.append("Esta verificación automática no es una prueba: un defecto chico, como una mancha, "
                  "puede desaparecer sin que ningún detector lo note.")
    return "\n\n".join(lineas)
