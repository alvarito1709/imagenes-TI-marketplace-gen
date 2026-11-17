"""Interfaz Gradio del restaurador de fotos de productos.

La lógica vive en src/; acá solo se arma la pantalla y se conectan los pasos.
Uso:  uv run app.py      (en Colab: ver colab/ejecutar_en_colab.ipynb)
"""
import logging
import time

import gradio as gr
from PIL import Image

from src import analysis, clasico, diffusion, escena, honesty, imagenes, segmentacion

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


def procesar(imagen, intensidad, corregir_luz=True, progreso=gr.Progress()):
    if imagen is None:
        raise gr.Error("Primero subí una foto de tu producto.")
    try:
        progreso(0.1, desc="Preparando la foto...")
        original = imagenes.preparar(imagen)

        # Corrección clásica de luz ANTES de la difusión: es lo que el modelo no resuelve solo.
        entrada_difusion = clasico.corregir_iluminacion(original) if corregir_luz else original

        progreso(0.3, desc="Reconstruyendo detalles con un modelo de difusión...")
        resultado = diffusion.restaurar(entrada_difusion, intensidad)

        progreso(0.7, desc="Comparando los objetos de la foto antes y después...")
        objetos_antes = analysis.detectar_objetos(original)
        objetos_despues = analysis.detectar_objetos(resultado)
        comparacion = analysis.comparar_objetos(objetos_antes, objetos_despues)

        progreso(0.9, desc="Armando el informe...")
        mapa = honesty.mapa_de_diferencias(original, resultado)
        base_clasica = clasico.filtro_bilateral(entrada_difusion)  # misma luz que recibió el modelo
        informe = honesty.generar_informe(
            comparacion, intensidad, honesty.porcentaje_cambiado(original, resultado), corregir_luz
        )
    except Exception as error:
        # El detalle técnico va a la consola; la persona ve un mensaje simple.
        registro.exception("Falló el procesamiento")
        raise gr.Error(
            "No pude procesar la foto. Probá con una más chica o en formato JPG."
        ) from error

    # El último par (original, resultado) queda guardado para el paso opcional de cambiar el fondo.
    return (_comparador(original, resultado), mapa, base_clasica,
            f"{honesty.AVISO_IA}\n\n{informe}", (original, resultado))


def cambiar_fondo(ultimo, fondo, progreso=gr.Progress()):
    """Paso 2 (opcional): cambia el fondo de la foto ya restaurada, sin tocar los píxeles del producto.

    Los modelos de este paso (BiRefNet y el inpainting) se cargan recién acá, la primera vez que
    se usa: quien no lo necesita no paga ni memoria ni tiempo.
    """
    if ultimo is None:
        raise gr.Error("Primero mejorá una foto en el paso 1.")
    original, restaurada = ultimo
    try:
        progreso(0.1, desc="Separando el producto del fondo (la primera vez carga el modelo)...")
        binaria = segmentacion.limpiar_mascara(segmentacion.obtener_mascara(restaurada))
        alfa = segmentacion.alfa_para_composicion(binaria)

        usa_ia = fondo != escena.FONDO_SIN_IA
        if usa_ia:
            progreso(0.4, desc="Generando el fondo nuevo con IA (la primera vez carga el modelo)...")
            fondo_img = escena.generar_fondo(restaurada.size, fondo)
        else:
            fondo_img = escena.fondo_liso(restaurada.size)
        resultado = escena.componer(restaurada, fondo_img, alfa)

        progreso(0.9, desc="Armando el informe...")
        mascara_vista = Image.fromarray(binaria).convert("L").point(lambda v: 255 if v else 0)
        informe = honesty.generar_informe_fondo(
            fondo, usa_ia,
            honesty.porcentaje_producto_modificado(restaurada, resultado, binaria),
            honesty.porcentaje_producto_modificado(original, resultado, binaria),
        )
    except Exception as error:
        registro.exception("Falló el cambio de fondo")
        raise gr.Error(
            "No pude cambiar el fondo. Probá con otra foto o con el fondo blanco liso."
        ) from error
    return resultado, mascara_vista, informe


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
        corregir_luz = gr.Checkbox(
            value=True,
            label="Corregir la iluminación primero (método clásico)",
            info="Ayuda con fotos oscuras. La IA de este paso limpia ruido y desenfoque, no arregla la luz. El fondo se cambia en el paso 2 (opcional).",
        )
        boton = gr.Button("Mejorar foto", variant="primary")
        if HAY_SLIDER:
            comparador = gr.ImageSlider(label="Antes / Después")
        else:
            comparador = gr.Gallery(label="Antes / Después", columns=2)
        mapa = gr.Image(label="Dónde cambió más (rojo = cambió mucho)")
        base = gr.Image(label="Solo filtro clásico (sin IA), para comparar")
        informe = gr.Markdown()
        ultimo = gr.State()   # (original, resultado) del paso 1, para el paso 2
        boton.click(procesar, [entrada, intensidad, corregir_luz], [comparador, mapa, base, informe, ultimo])

        with gr.Accordion("Paso 2 (opcional): cambiar el fondo", open=False):
            gr.Markdown(
                "Se aplica sobre la foto del paso 1. **Tu producto no se toca: la IA solo genera el fondo.** "
                "La primera vez tarda más porque carga otros modelos."
            )
            fondo = gr.Radio(
                escena.OPCIONES_FONDO, value=escena.OPCIONES_FONDO[0], label="Fondo",
                info="Los primeros tres los genera una IA. El último es un blanco liso, sin IA.",
            )
            boton_fondo = gr.Button("Cambiar fondo")
            foto_fondo = gr.Image(label="Foto con el fondo nuevo")
            mascara = gr.Image(label="Qué detectó el modelo como producto (blanco)")
            informe_fondo = gr.Markdown()
            boton_fondo.click(cambiar_fondo, [ultimo, fondo], [foto_fondo, mascara, informe_fondo])
    return pantalla


demo = construir_pantalla()


def iniciar(share: bool = False) -> None:
    """Precarga los modelos y lanza la app. En Colab: iniciar(share=True)."""
    precargar_modelos()
    demo.launch(share=share)


if __name__ == "__main__":
    iniciar()
