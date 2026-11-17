"""Interfaz Gradio del restaurador de fotos de productos.

La lógica vive en src/; acá solo se arma la pantalla y se conectan los pasos.
Uso:  uv run app.py      (en Colab: ver colab/ejecutar_en_colab.ipynb)
"""
import logging
import time

import gradio as gr

from src import analysis, clasico, diffusion, honesty, imagenes

logging.basicConfig(level=logging.INFO)
registro = logging.getLogger(__name__)


def precargar_modelos() -> None:
    """Carga los modelos al arrancar, para que el primer clic no parezca colgado.

    La primera vez descarga varios GB (puede tardar unos 4 minutos).
    """
    t0 = time.time()
    registro.info("Cargando el modelo de difusión (la primera vez descarga varios GB)...")
    diffusion.cargar_pipeline()
    registro.info("Cargando el detector de objetos...")
    analysis.cargar_detector()
    registro.info("Modelos listos en %.0f s", time.time() - t0)


def procesar(imagen, intensidad, progreso=gr.Progress()):
    if imagen is None:
        raise gr.Error("Primero subí una foto de tu producto.")
    try:
        progreso(0.1, desc="Preparando la foto...")
        original = imagenes.preparar(imagen)

        progreso(0.3, desc="Reconstruyendo detalles con un modelo de difusión...")
        resultado = diffusion.restaurar(original, intensidad)

        progreso(0.7, desc="Comparando los objetos de la foto antes y después...")
        objetos_antes = analysis.detectar_objetos(original)
        objetos_despues = analysis.detectar_objetos(resultado)
        comparacion = analysis.comparar_objetos(objetos_antes, objetos_despues)

        progreso(0.9, desc="Armando el informe...")
        mapa = honesty.mapa_de_diferencias(original, resultado)
        base_clasica = clasico.filtro_bilateral(original)
        informe = honesty.generar_informe(
            comparacion, intensidad, honesty.porcentaje_cambiado(original, resultado)
        )
    except Exception as error:
        # El detalle técnico va a la consola; la persona ve un mensaje simple.
        registro.exception("Falló el procesamiento")
        raise gr.Error(
            "No pude procesar la foto. Probá con una más chica o en formato JPG."
        ) from error

    return _comparador(original, resultado), mapa, base_clasica, f"{honesty.AVISO_IA}\n\n{informe}"


def _comparador(original, resultado):
    """Formato de salida según el componente que haya disponible (ver construir_pantalla)."""
    if HAY_SLIDER:
        return (original, resultado)
    return [(original, "Antes"), (resultado, "Después (reconstruida por IA)")]


# La API de Gradio cambia entre versiones: si no existe ImageSlider, mostramos
# las dos fotos en una galería para no romper la app.
HAY_SLIDER = hasattr(gr, "ImageSlider")


def construir_pantalla() -> gr.Blocks:
    with gr.Blocks(title="Restaurador de fotos de productos") as pantalla:
        gr.Markdown(
            "# Restaurador de fotos de productos\n"
            "Subí la foto de lo que vendés y mirá qué cambia. "
            "**Los detalles nuevos los genera una IA: revisalos antes de publicar.**"
        )
        entrada = gr.Image(type="pil", label="Tu foto")
        intensidad = gr.Slider(
            0.0, 1.0, value=0.3, step=0.05,
            label="Cuánto puede cambiar el modelo mi foto",
            info="Más bajo = más fiel a tu foto. Más alto = el modelo inventa más.",
        )
        boton = gr.Button("Mejorar foto", variant="primary")
        if HAY_SLIDER:
            comparador = gr.ImageSlider(label="Antes / Después")
        else:
            comparador = gr.Gallery(label="Antes / Después", columns=2)
        mapa = gr.Image(label="Dónde cambió más (rojo = cambió mucho)")
        base = gr.Image(label="Mejora clásica (sin IA), para comparar")
        informe = gr.Markdown()
        boton.click(procesar, [entrada, intensidad], [comparador, mapa, base, informe])
    return pantalla


demo = construir_pantalla()


def iniciar(share: bool = False) -> None:
    """Precarga los modelos y lanza la app. En Colab: iniciar(share=True)."""
    precargar_modelos()
    demo.launch(share=share)


if __name__ == "__main__":
    iniciar()
