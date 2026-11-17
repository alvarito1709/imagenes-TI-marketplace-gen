"""Prueba rápida del modelo SIN interfaz (para correr en Colab con GPU).

Uso:
    python scripts/prueba_difusion.py                  # imagen sintética
    python scripts/prueba_difusion.py foto.jpg 0.3     # tu foto e intensidad

Guarda en salidas/ la entrada degradada y el resultado lado a lado.
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import diffusion, imagenes  # noqa: E402


def imagen_sintetica() -> Image.Image:
    """Formas simples con ruido y desenfoque, para probar que el modelo carga."""
    base = Image.new("RGB", (512, 512), (200, 190, 170))
    dibujo = ImageDraw.Draw(base)
    dibujo.rectangle((120, 150, 390, 420), fill=(40, 80, 160))
    dibujo.ellipse((200, 60, 310, 170), fill=(220, 180, 140))
    base = base.filter(ImageFilter.GaussianBlur(2))
    ruido = np.random.default_rng(0).normal(0, 18, (512, 512, 3))
    return Image.fromarray(np.clip(np.asarray(base) + ruido, 0, 255).astype("uint8"))


def main() -> None:
    ruta = sys.argv[1] if len(sys.argv) > 1 else None
    intensidad = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

    entrada = imagenes.preparar(Image.open(ruta) if ruta else imagen_sintetica())
    print(f"Entrada: {entrada.size}, intensidad={intensidad}")

    t0 = time.time()
    diffusion.cargar_pipeline()
    print(f"Modelo cargado en {time.time() - t0:.0f} s")

    t0 = time.time()
    resultado = diffusion.restaurar(entrada, intensidad)
    print(f"Procesado en {time.time() - t0:.0f} s")

    lado_a_lado = Image.new("RGB", (entrada.width * 2, entrada.height))
    lado_a_lado.paste(entrada, (0, 0))
    lado_a_lado.paste(resultado, (entrada.width, 0))
    Path("salidas").mkdir(exist_ok=True)
    lado_a_lado.save("salidas/prueba_difusion.png")
    print("Guardado en salidas/prueba_difusion.png")


if __name__ == "__main__":
    main()
