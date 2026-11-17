"""Prueba rápida de RT-DETR sin interfaz (Colab o local, funciona también en CPU).

Uso:  python scripts/prueba_analisis.py foto.jpg [foto_procesada.jpg]
Imprime los objetos detectados y, si das dos fotos, qué cambió entre ellas.
"""
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import analysis, honesty  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Uso: python scripts/prueba_analisis.py foto.jpg [foto_procesada.jpg]")
    imagenes = [Image.open(ruta) for ruta in sys.argv[1:3]]

    t0 = time.time()
    analysis.cargar_detector()
    print(f"Detector cargado en {time.time() - t0:.0f} s")

    detecciones = [analysis.detectar_objetos(i) for i in imagenes]
    for ruta, objetos in zip(sys.argv[1:3], detecciones):
        print(f"{ruta}: {[(o['label'], round(o['score'], 2)) for o in objetos] or 'nada detectado'}")

    if len(imagenes) == 2:
        comparacion = analysis.comparar_objetos(*detecciones)
        print(comparacion)
        print(honesty.generar_informe(comparacion, 0.3, honesty.porcentaje_cambiado(*imagenes)))


if __name__ == "__main__":
    main()
