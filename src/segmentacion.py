"""Segmentación del producto (paso 2 opcional de la app: cambiar el fondo): separa "lo que se vende" del fondo.

Un modelo (BiRefNet) dibuja una máscara: blanco = producto, negro = fondo.
Con esa máscara la IA solo toca el fondo y los píxeles del producto se pegan
tal cual estaban en la foto original (ver escena.py).

Las funciones de abajo del modelo (binarizar, quitar_islas, etc.) son OpenCV
puro: se prueban sin GPU y sin descargar nada.
"""
import cv2
import numpy as np
from PIL import Image

MODELO_SEGMENTACION = "ZhengPeng7/BiRefNet"  # licencia MIT
RESOLUCION = 1024                            # tamaño de entrada con el que se entrenó BiRefNet
_MEDIA = (0.485, 0.456, 0.406)               # normalización ImageNet
_DESVIO = (0.229, 0.224, 0.225)

UMBRAL = 128            # sobre 255: desde acá un píxel cuenta como producto
AREA_MINIMA = 0.02      # una isla de menos del 2% del producto se descarta (ruido de la máscara)
EROSION_PX = 1          # se achica el borde del producto para no arrastrar fondo viejo (halo)
DIFUMINADO_PX = 1       # suaviza el borde al pegar, para que no se vea recortado con tijera
EXPANSION_PX = 6        # cuánto invade el inpainting dentro del borde del producto
HUECO_MAXIMO = 0.005    # solo se tapan huecos de menos del 0.5% del producto (botones, ojales)
ESCALA_REFERENCIA = 768 # los píxeles de arriba están pensados para una foto de 768 px de lado

_modelo = None  # caché: el modelo se carga una sola vez


def _dispositivo() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def cargar_modelo():
    """Carga BiRefNet UNA sola vez (import perezoso: los tests no necesitan torch)."""
    global _modelo
    if _modelo is not None:
        return _modelo
    from transformers import AutoModelForImageSegmentation

    modelo = AutoModelForImageSegmentation.from_pretrained(MODELO_SEGMENTACION, trust_remote_code=True)
    # Las versiones nuevas de transformers cargan los pesos en el tipo guardado (float16),
    # y la entrada va en float32: se fuerza float32 para que coincidan.
    modelo.float().to(_dispositivo()).eval()
    _modelo = modelo
    return _modelo


def obtener_mascara(imagen: Image.Image) -> Image.Image:
    """Máscara "L" (0-255, mismo tamaño que la foto): más claro = más seguro que es producto."""
    import torch

    modelo = cargar_modelo()
    rgb = imagen.convert("RGB")
    chica = np.asarray(rgb.resize((RESOLUCION, RESOLUCION), Image.BILINEAR), dtype=np.float32) / 255.0
    chica = (chica - np.array(_MEDIA, dtype=np.float32)) / np.array(_DESVIO, dtype=np.float32)
    entrada = torch.from_numpy(chica.transpose(2, 0, 1)).unsqueeze(0)
    entrada = entrada.to(_dispositivo()).to(next(modelo.parameters()).dtype)  # mismo tipo que los pesos

    with torch.no_grad():
        prediccion = modelo(entrada)[-1].sigmoid().cpu()[0, 0].numpy()  # BiRefNet devuelve varias salidas; la última es la fina
    mascara = Image.fromarray((prediccion * 255).astype(np.uint8), mode="L")
    return mascara.resize(rgb.size, Image.BILINEAR)


def binarizar(mascara: Image.Image, umbral: int = UMBRAL) -> np.ndarray:
    """Máscara booleana (alto x ancho): True = producto."""
    return np.asarray(mascara.convert("L")) >= umbral


def quitar_islas(binaria: np.ndarray, area_minima: float = AREA_MINIMA) -> np.ndarray:
    """Descarta manchitas sueltas que la máscara marcó como producto sin serlo.

    Se conservan las zonas conectadas cuyo tamaño es al menos `area_minima` (fracción)
    de la más grande: así un producto en dos partes (un par de aros) no pierde una.
    """
    sin_islas = np.zeros_like(binaria)
    cantidad, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(binaria.astype(np.uint8), connectivity=8)
    if cantidad <= 1:
        return binaria.copy()
    areas = estadisticas[1:, cv2.CC_STAT_AREA]
    for indice, area in enumerate(areas, start=1):
        if area >= areas.max() * area_minima:
            sin_islas[etiquetas == indice] = True
    return sin_islas


def rellenar_huecos(binaria: np.ndarray, area_maxima: float = HUECO_MAXIMO) -> np.ndarray:
    """Tapa agujeros CHICOS adentro del producto (un botón que la máscara dejó como fondo).

    Un hueco grande no se toca: el espacio entre dos correas o dentro de un lazo es
    fondo de verdad, y taparlo dejaría el sillón pegado al producto.
    """
    fondo = (~binaria).astype(np.uint8)
    cantidad, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(fondo, connectivity=4)
    limite = binaria.sum() * area_maxima
    alto, ancho = binaria.shape
    lleno = binaria.copy()
    for i in range(1, cantidad):
        x, y, w, h, area = estadisticas[i]
        toca_borde = x == 0 or y == 0 or x + w == ancho or y + h == alto
        if not toca_borde and area <= limite:
            lleno[etiquetas == i] = True
    return lleno


def _escala(binaria: np.ndarray) -> float:
    """Los píxeles de borde crecen con la foto (en una de 1600 px un borde de 2 px no se ve). Nunca menos de 1."""
    return max(1.0, max(binaria.shape) / ESCALA_REFERENCIA)


def _kernel(radio: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radio + 1, 2 * radio + 1))


def limpiar_mascara(mascara: Image.Image) -> np.ndarray:
    """Máscara del modelo → máscara booleana limpia (sin islas ni agujeros)."""
    return rellenar_huecos(quitar_islas(binarizar(mascara)))


def alfa_para_composicion(binaria: np.ndarray, erosion_px: int = EROSION_PX,
                          difuminado_px: int = DIFUMINADO_PX) -> np.ndarray:
    """Transparencia (float 0-1) con la que se pega el producto sobre el fondo nuevo.

    Se erosiona un poco el borde (para no arrastrar píxeles del fondo viejo) y se
    difumina (para que la transición no se vea recortada).
    """
    escala = _escala(binaria)
    erosion_px, difuminado_px = round(erosion_px * escala), difuminado_px * escala
    mascara = binaria.astype(np.uint8) * 255
    if erosion_px > 0:
        mascara = cv2.erode(mascara, _kernel(erosion_px))
    if difuminado_px > 0:
        mascara = cv2.GaussianBlur(mascara, (0, 0), difuminado_px)
    return mascara.astype(np.float32) / 255.0


def mascara_para_inpainting(binaria: np.ndarray, expansion_px: int = EXPANSION_PX) -> Image.Image:
    """Máscara "L" para el inpainting: blanco = lo que la IA puede generar (el fondo).

    El producto queda protegido, pero la zona a generar entra `expansion_px` en su borde:
    así el fondo nuevo llega hasta abajo del borde y no queda un anillo del fondo viejo.
    Los píxeles de ese borde se pisan después con los originales.
    """
    expansion_px = round(expansion_px * _escala(binaria))
    producto = binaria.astype(np.uint8) * 255
    if expansion_px > 0:
        producto = cv2.erode(producto, _kernel(expansion_px))
    fondo = 255 - producto
    return Image.fromarray(fondo, mode="L")
