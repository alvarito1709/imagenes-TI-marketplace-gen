"""Interfaz Gradio del mejorador de escena para fotos de productos.

El producto se conserva píxel por píxel; la IA solo genera el fondo.
La lógica vive en src/; acá solo se arma la pantalla y se conectan los pasos.
Uso:  uv run app.py      (en Colab/Kaggle: ver los cuadernos de colab/)

El enfoque anterior (retocar el producto con img2img + ControlNet) quedó en
src/diffusion.py como plan B; esta app ya no lo usa.
"""
import logging
import time

import gradio as gr
from PIL import Image

from src import analysis, clasico, escena, honesty, imagenes, segmentacion

logging.basicConfig(level=logging.INFO)
registro = logging.getLogger(__name__)


def precargar_modelos() -> None:
    """Carga los modelos al arrancar, para que el primer clic no parezca colgado.

    La primera vez descarga varios GB (puede tardar unos minutos).
    """
    t0 = time.time()
    registro.info("Cargando el modelo de segmentación (BiRefNet)...")
    segmentacion.cargar_modelo()
    registro.info("Cargando el modelo de inpainting (la primera vez descarga varios GB)...")
    escena.cargar_inpainting()
    registro.info("Cargando el detector de objetos...")
    analysis.cargar_detector()
    registro.info("Modelos listos en %.0f s", time.time() - t0)


def procesar(imagen, fondo, corregir_luz=True, progreso=gr.Progress()):
    if imagen is None:
        raise gr.Error("Primero subí una foto de tu producto.")
    try:
        progreso(0.1, desc="Preparando la foto...")
        original = imagenes.preparar(imagen)

        progreso(0.2, desc="Separando el producto del fondo...")
        binaria = segmentacion.limpiar_mascara(segmentacion.obtener_mascara(original))
        alfa = segmentacion.alfa_para_composicion(binaria)

        # La luz se corrige SOLO sobre los píxeles del producto, con un método clásico (no IA).
        producto = clasico.corregir_iluminacion(original) if corregir_luz else original

        usa_ia = fondo != escena.FONDO_SIN_IA
        if usa_ia:
            progreso(0.4, desc="Generando el fondo nuevo con IA...")
            fondo_img = escena.generar_fondo(original, segmentacion.mascara_para_inpainting(binaria), fondo)
        else:
            fondo_img = escena.fondo_liso(original.size)
        resultado = escena.componer(producto, fondo_img, alfa)

        progreso(0.8, desc="Comparando los objetos de la foto antes y después...")
        comparacion = analysis.comparar_objetos(
            analysis.detectar_objetos(original), analysis.detectar_objetos(resultado)
        )

        progreso(0.9, desc="Armando el informe...")
        mapa = honesty.mapa_de_diferencias(original, resultado)
        base = escena.componer(producto, escena.fondo_liso(original.size), alfa)  # sin IA, misma luz
        mascara_vista = Image.fromarray(binaria).convert("L").point(lambda v: 255 if v else 0)
        informe = honesty.generar_informe_escena(
            fondo, usa_ia,
            honesty.porcentaje_producto_modificado(original, resultado, binaria),
            corregir_luz, comparacion,
        )
    except Exception as error:
        # El detalle técnico va a la consola; la persona ve un mensaje simple.
        registro.exception("Falló el procesamiento")
        raise gr.Error(
            "No pude procesar la foto. Probá con una más chica o en formato JPG."
        ) from error

    aviso = honesty.AVISO_ESCENA if usa_ia else ""
    return _comparador(original, resultado, usa_ia), mapa, mascara_vista, base, f"{aviso}\n\n{informe}"


def _comparador(original, resultado, usa_ia=True):
    """Formato de salida según el componente que haya disponible (ver construir_pantalla)."""
    etiqueta = "Después (fondo generado por IA)" if usa_ia else "Después (fondo liso, sin IA)"
    if HAY_SLIDER:
        return (original, resultado)
    return [(original, "Antes"), (resultado, etiqueta)]


# La API de Gradio cambia entre versiones: si no existe ImageSlider, mostramos
# las dos fotos en una galería para no romper la app.
HAY_SLIDER = hasattr(gr, "ImageSlider")


def construir_pantalla() -> gr.Blocks:
    with gr.Blocks(title="Mejorador de fotos de productos") as pantalla:
        gr.Markdown(
            "# Mejorador de fotos de productos\n"
            "Subí la foto de lo que vendés y elegí un fondo. "
            "**Tu producto no se toca: la IA solo genera el fondo.**"
        )
        entrada = gr.Image(type="pil", label="Tu foto")
        fondo = gr.Radio(
            escena.OPCIONES_FONDO, value=escena.OPCIONES_FONDO[0],
            label="Fondo",
            info="Los primeros tres los genera una IA. El último es un blanco liso, sin IA, para comparar.",
        )
        corregir_luz = gr.Checkbox(
            value=True,
            label="Corregir la iluminación del producto (método clásico)",
            info="Ayuda con fotos oscuras. Modifica los píxeles del producto: el informe te dice cuánto.",
        )
        boton = gr.Button("Mejorar foto", variant="primary")
        if HAY_SLIDER:
            comparador = gr.ImageSlider(label="Antes / Después")
        else:
            comparador = gr.Gallery(label="Antes / Después", columns=2)
        mapa = gr.Image(label="Dónde cambió (rojo = cambió mucho)")
        mascara = gr.Image(label="Qué detectó el modelo como producto (blanco)")
        base = gr.Image(label="Solo fondo blanco liso (sin IA), para comparar")
        informe = gr.Markdown()
        boton.click(procesar, [entrada, fondo, corregir_luz], [comparador, mapa, mascara, base, informe])
    return pantalla


demo = construir_pantalla()


def iniciar(share: bool = False) -> None:
    """Precarga los modelos y lanza la app. En Colab/Kaggle: iniciar(share=True)."""
    precargar_modelos()
    demo.launch(share=share)


if __name__ == "__main__":
    iniciar()
