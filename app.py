"""Interfaz Gradio del restaurador de fotos de productos.

La lógica vive en src/; acá solo se arma la pantalla y se conectan los pasos.
ESTADO: esqueleto. Los pasos de src/ todavía lanzan NotImplementedError.
"""
import gradio as gr

from src import analysis, clasico, diffusion, honesty, imagenes


def procesar(imagen, intensidad, progreso=gr.Progress()):
    if imagen is None:
        raise gr.Error("Primero subí una foto de tu producto.")
    try:
        progreso(0.1, desc="Preparando la foto...")
        original = imagenes.preparar(imagen)

        progreso(0.4, desc="Reconstruyendo detalles con un modelo de difusión...")
        resultado = diffusion.restaurar(original, intensidad)

        progreso(0.7, desc="Comparando objetos antes y después...")
        objetos_antes = analysis.detectar_objetos(original)
        objetos_despues = analysis.detectar_objetos(resultado)
        comparacion = analysis.comparar_objetos(objetos_antes, objetos_despues)

        mapa = honesty.mapa_de_diferencias(original, resultado)
        base_clasica = clasico.filtro_bilateral(original)
        informe = honesty.generar_informe(comparacion, intensidad)
    except NotImplementedError as error:
        raise gr.Error(f"Esta parte todavía no está implementada: {error}") from error
    except Exception as error:
        raise gr.Error(
            "No pude procesar la foto. Probá con una más chica o en formato JPG."
        ) from error

    return (original, resultado), mapa, base_clasica, f"{honesty.AVISO_IA}\n\n{informe}"


with gr.Blocks(title="Restaurador de fotos de productos") as demo:
    gr.Markdown(
        "# Restaurador de fotos de productos\n"
        "Subí la foto de lo que vendés y mirá qué cambia. "
        "**Los detalles nuevos los genera una IA: revisalos antes de publicar.**"
    )
    entrada = gr.Image(type="pil", label="Tu foto")
    intensidad = gr.Slider(
        0.0, 1.0, value=0.3, step=0.05,
        label="Cuánto puede cambiar el modelo mi foto",
    )
    boton = gr.Button("Mejorar foto", variant="primary")
    comparador = gr.ImageSlider(label="Antes / Después")  # revisar API según versión de Gradio
    mapa = gr.Image(label="Dónde cambió más")
    base = gr.Image(label="Mejora clásica (sin IA), para comparar")
    informe = gr.Markdown()
    boton.click(procesar, [entrada, intensidad], [comparador, mapa, base, informe])

if __name__ == "__main__":
    demo.launch()
