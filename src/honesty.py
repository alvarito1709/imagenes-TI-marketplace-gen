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

AVISO_ESCENA = (
    "ℹ️ **Qué es real y qué es inventado:** el producto es el de tu foto, píxel por píxel. "
    "Lo único que generó una IA es el fondo. Si ves que el producto \"flota\" o le falta sombra, "
    "es porque el fondo es nuevo: usá la foto con criterio."
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
                    porcentaje: float | None = None,
                    iluminacion_corregida: bool = False) -> str:
    """Informe en lenguaje simple (Markdown) para la persona usuaria.

    `comparacion_objetos` viene de analysis.comparar_objetos, con las claves
    'aparecieron', 'desaparecieron' y 'sin_cambios' (listas de nombres de objetos).
    """
    lineas = [f"**Intensidad usada:** {intensidad:.2f} → {_nivel(intensidad)}."]

    if iluminacion_corregida:
        lineas.append("**Iluminación:** se corrigió con un método clásico (no es IA): reparte mejor la luz "
                      "que ya tenía tu foto, pero no recupera lo que la cámara no registró y puede resaltar el ruido.")

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


def porcentaje_producto_modificado(original: Image.Image, resultado: Image.Image,
                                   binaria: np.ndarray, borde_px: int = 8) -> float:
    """Porcentaje (0-100) de píxeles DEL PRODUCTO que cambiaron de forma visible.

    Es la métrica clave de este enfoque: sin corrección de luz tiene que dar 0%,
    porque los píxeles del producto se pegan sin tocar. Se excluye un borde fino
    (`borde_px`, un poco más que la erosión + difuminado de segmentacion.alfa_para_composicion)
    porque ahí la mezcla con el fondo nuevo es esperable.
    """
    interior = cv2.erode(binaria.astype(np.uint8), np.ones((2 * borde_px + 1,) * 2, np.uint8)).astype(bool)
    if not interior.any():
        return 0.0
    dif = _diferencia(original, resultado)
    return float((dif[interior] > DIFERENCIA_VISIBLE).mean() * 100)


def generar_informe_escena(fondo: str, usa_ia: bool, producto_modificado: float,
                           iluminacion_corregida: bool = False,
                           comparacion_objetos: dict | None = None) -> str:
    """Informe en lenguaje simple (Markdown) para el enfoque "producto real + fondo nuevo"."""
    if usa_ia:
        lineas = [f"**Fondo:** {fondo}, generado por una IA. Es lo único inventado de la foto."]
    else:
        lineas = [f"**Fondo:** {fondo}. No usa IA: es un color liso, para que compares si el fondo "
                  "generado realmente aporta algo."]

    if producto_modificado < 0.05:
        lineas.append("🟢 **Tu producto:** 0% de sus píxeles se modificó. Logos, botones, estampados y "
                      "detalles son los de tu foto real.")
    elif iluminacion_corregida:
        lineas.append(f"🟡 **Tu producto:** {producto_modificado:.1f}% de sus píxeles cambió, por la corrección "
                      "de iluminación (método clásico, no IA). Ajusta brillo y contraste, no inventa detalles.")
    else:
        lineas.append(f"🔴 **Tu producto:** {producto_modificado:.1f}% de sus píxeles cambió y no debería. "
                      "Revisá el resultado.")

    if iluminacion_corregida:
        lineas.append("**Iluminación:** se corrigió con un método clásico: reparte mejor la luz que ya tenía "
                      "tu foto, pero no recupera lo que la cámara no registró y puede resaltar el ruido.")

    if comparacion_objetos:
        aparecieron = comparacion_objetos.get("aparecieron", [])
        desaparecieron = comparacion_objetos.get("desaparecieron", [])
        if aparecieron:
            lineas.append(f"🔴 **Aparecieron objetos que no estaban:** {', '.join(aparecieron)}. "
                          "El fondo generado puede haberlos inventado.")
        if desaparecieron:
            lineas.append(f"🔴 **Desaparecieron objetos que sí estaban:** {', '.join(desaparecieron)}. "
                          "Verificá que la máscara no haya recortado parte del producto.")

    lineas.append("**Límites:** la máscara puede fallar en bordes finos (flecos, tiras, tul) y el producto "
                  "no proyecta sombra sobre el fondo nuevo. Mirá el borde antes de publicar.")
    return "\n\n".join(lineas)


def generar_informe_fondo(fondo: str, usa_ia: bool, cambio_en_este_paso: float, cambio_total: float) -> str:
    """Informe (Markdown) del paso opcional "cambiar el fondo" que se aplica sobre la foto ya restaurada.

    `cambio_en_este_paso`: % de píxeles del producto que cambiaron al pegarlo sobre el fondo nuevo (debe ser 0).
    `cambio_total`: % de píxeles del producto que cambiaron respecto de la foto ORIGINAL de la persona
    (incluye lo que hizo la restauración del paso 1).
    """
    if usa_ia:
        lineas = [f"**Fondo:** {fondo}, generado por una IA sin ver tu producto. Es lo único inventado en este paso."]
    else:
        lineas = [f"**Fondo:** {fondo}. No usa IA: es un color liso."]

    if cambio_en_este_paso < 0.05:
        lineas.append("🟢 **En este paso el producto no se tocó:** sus píxeles son los de la foto del paso 1.")
    else:
        lineas.append(f"🔴 **En este paso cambió {cambio_en_este_paso:.1f}% del producto y no debería.** "
                      "Revisá el resultado.")

    if cambio_total < 0.05:
        lineas.append("🟢 Respecto de tu foto original, el producto tampoco cambió.")
    else:
        lineas.append(f"🟡 Respecto de tu foto original, {cambio_total:.1f}% de los píxeles del producto cambió "
                      "por el paso 1 (restauración y luz). Mirá el mapa del paso 1.")

    lineas.append("**Límites:** el recorte puede incluir partes que no son del producto (una mano, un trozo de "
                  "sillón) o comerse bordes finos como flecos y tiras, y el producto no proyecta sombra sobre el "
                  "fondo nuevo. Mirá el borde antes de publicar.")
    return "\n\n".join(lineas)
