"""Evaluación honesta: ¿la difusión se acerca más a la foto buena que un filtro clásico?

Parte de una foto buena, la degrada (desenfoque + ruido), la restaura de varias formas y mide
PSNR y SSIM contra la foto buena. Necesita GPU para la difusión (Colab o Kaggle).

Uso:  python scripts/evaluar_degradacion.py foto_buena.jpg [desenfoque] [ruido]
Guarda las imágenes y resultados.md en salida_evaluacion/.
"""
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import clasico, diffusion, imagenes, metricas  # noqa: E402

INTENSIDADES = (0.1, 0.3, 0.5)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Uso: python scripts/evaluar_degradacion.py foto_buena.jpg [desenfoque] [ruido]")
    ruta = Path(sys.argv[1])
    if not ruta.is_file():
        sys.exit(f"No encuentro la foto '{ruta}'. Subí una foto NÍTIDA a esa ruta (y hacé git push/pull si corrés en Kaggle o Colab).")
    desenfoque = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    ruido = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    salida = Path("salida_evaluacion")
    salida.mkdir(exist_ok=True)

    buena = imagenes.preparar(Image.open(ruta))
    degradada = metricas.degradar(buena, desenfoque, ruido)
    buena.save(salida / "0_original.png")
    degradada.save(salida / "1_degradada.png")

    candidatos = [("Foto degradada (sin hacer nada)", degradada)]
    candidatos.append(("Filtro bilateral (clásico)", clasico.filtro_bilateral(degradada)))
    candidatos.append(("Filtro mediano 5x5 (clásico)", clasico.filtro_mediano(degradada, 5)))

    t0 = time.time()
    diffusion.cargar_pipeline()
    print(f"Modelo cargado en {time.time() - t0:.0f} s")
    for intensidad in INTENSIDADES:
        candidatos.append((f"Difusión, intensidad {intensidad}", diffusion.restaurar(degradada, intensidad)))

    filas = []
    for indice, (nombre, imagen) in enumerate(candidatos, start=2):
        imagen.save(salida / f"{indice}_{nombre.split(',')[0].split('(')[0].strip().replace(' ', '_')}.png")
        filas.append((nombre, metricas.psnr(buena, imagen), metricas.ssim(buena, imagen)))

    tabla = metricas.tabla_markdown(filas)
    texto = (f"Foto: {ruta.name} | desenfoque σ={desenfoque} | ruido σ={ruido}\n\n{tabla}\n\n"
             "Las métricas miden parecido con la foto buena, no si algo es auténtico.\n")
    (salida / "resultados.md").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
