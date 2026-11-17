"""Prueba rápida de la escena nueva sin interfaz (necesita GPU: Colab o Kaggle).

Uso:  python scripts/prueba_escena.py foto.jpg [fondo]
`fondo` puede ser: "Blanco de estudio", "Madera clara", "Gris suave" (por defecto el primero).
Guarda en salida_escena/ la máscara, el resultado y las medidas, e imprime el informe.
"""
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import escena, honesty, imagenes, segmentacion  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Uso: python scripts/prueba_escena.py foto.jpg [fondo]")
    ruta = Path(sys.argv[1])
    fondo = sys.argv[2] if len(sys.argv) > 2 else escena.OPCIONES_FONDO[0]
    salida = Path("salida_escena")
    salida.mkdir(exist_ok=True)

    original = imagenes.preparar(Image.open(ruta))
    print(f"Foto: {ruta.name} → {original.size}")

    t0 = time.time()
    mascara_cruda = segmentacion.obtener_mascara(original)
    print(f"Segmentación: {time.time() - t0:.1f} s (incluye la carga del modelo la primera vez)")
    binaria = segmentacion.limpiar_mascara(mascara_cruda)
    print(f"El producto ocupa {binaria.mean() * 100:.0f}% de la foto")
    alfa = segmentacion.alfa_para_composicion(binaria)

    t0 = time.time()
    fondo_ia = escena.generar_fondo(original.size, fondo)
    print(f"Inpainting '{fondo}': {time.time() - t0:.1f} s")
    resultado = escena.componer(original, fondo_ia, alfa)

    mascara_cruda.save(salida / f"{ruta.stem}_mascara.png")
    resultado.save(salida / f"{ruta.stem}_{fondo.replace(' ', '_')}.png")
    escena.componer(original, escena.fondo_liso(original.size), alfa).save(salida / f"{ruta.stem}_blanco_liso.png")
    honesty.mapa_de_diferencias(original, resultado).save(salida / f"{ruta.stem}_mapa.png")

    modificado = honesty.porcentaje_producto_modificado(original, resultado, binaria)
    print(f"Píxeles del producto modificados: {modificado:.2f}% (esperado: 0%)")
    print(honesty.generar_informe_escena(fondo, True, modificado))
    print(f"Archivos guardados en {salida}/")


if __name__ == "__main__":
    main()
