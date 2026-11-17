"""Métricas contra una referencia: ¿cuánto se acercó el modelo a la foto buena?

Con una foto real dañada NO hay referencia, así que estas métricas no se pueden calcular.
La forma honesta de usarlas es partir de una foto buena, degradarla nosotros (desenfoque + ruido),
restaurarla y comparar el resultado con la foto buena original.

Ninguna de estas métricas dice si algo es auténtico: miden parecido, no verdad. Un modelo puede
sacar un buen número y aun así haber inventado un logo.
"""
import cv2
import numpy as np
from PIL import Image


def _a_array(imagen: Image.Image) -> np.ndarray:
    return np.asarray(imagen.convert("RGB"), dtype=np.float64)


def _mismo_tamano(a: Image.Image, b: Image.Image) -> tuple[Image.Image, Image.Image]:
    if b.size != a.size:
        b = b.resize(a.size, Image.LANCZOS)
    return a, b


def psnr(referencia: Image.Image, imagen: Image.Image) -> float:
    """PSNR en dB: más alto = más parecida. Infinito si son idénticas."""
    referencia, imagen = _mismo_tamano(referencia, imagen)
    mse = np.mean((_a_array(referencia) - _a_array(imagen)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))


def ssim(referencia: Image.Image, imagen: Image.Image) -> float:
    """SSIM (0-1, Wang et al. 2004) promediado en los tres canales: 1 = idéntica.

    Compara estructura local (contraste y bordes), no solo diferencia de color.
    """
    referencia, imagen = _mismo_tamano(referencia, imagen)
    a, b = _a_array(referencia), _a_array(imagen)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def suave(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    valores = []
    for canal in range(3):
        x, y = a[..., canal], b[..., canal]
        mu_x, mu_y = suave(x), suave(y)
        var_x = suave(x * x) - mu_x ** 2
        var_y = suave(y * y) - mu_y ** 2
        cov = suave(x * y) - mu_x * mu_y
        mapa = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
        valores.append(mapa.mean())
    return float(np.mean(valores))


def degradar(imagen: Image.Image, desenfoque: float = 1.5, ruido: float = 15.0, semilla: int = 0) -> Image.Image:
    """Simula una foto de celular movida y con poca luz: desenfoque gaussiano + ruido gaussiano.

    `desenfoque` y `ruido` son desvíos estándar (en píxeles y en niveles de 0-255). La semilla fija el
    ruido: así la misma foto degradada se puede volver a generar y los métodos se comparan en igualdad.
    """
    arr = np.asarray(imagen.convert("RGB"), dtype=np.float32)
    if desenfoque > 0:
        arr = cv2.GaussianBlur(arr, (0, 0), desenfoque)
    if ruido > 0:
        arr = arr + np.random.default_rng(semilla).normal(0, ruido, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def tabla_markdown(filas: list[tuple[str, float, float]]) -> str:
    """Tabla (método, PSNR, SSIM) lista para pegar en el README."""
    lineas = ["| Método | PSNR (dB) ↑ | SSIM ↑ |", "|---|---|---|"]
    for nombre, valor_psnr, valor_ssim in filas:
        lineas.append(f"| {nombre} | {valor_psnr:.2f} | {valor_ssim:.3f} |")
    return "\n".join(lineas)
