# Restaurador de fotos de productos

> Limpia el ruido y el desenfoque de las fotos de tus productos con IA, y te muestra qué cambió para que no se oculte ningún defecto real.

**Trabajo Integrador N°2** — Procesamiento Digital de Imágenes e Introducción a Visión por Computadora · IFTS 24 · 2026

> **Estado:** en desarrollo. **Sub-tarea elegida: B (denoising/deblurring).** Hay además una exploración fuera del MVP (ver [Exploración fuera del MVP](#exploración-fuera-del-mvp-separar-producto-y-escena)).

## Persona usuaria

**Camila**, 28-35 años. Vende ropa y accesorios usados por marketplace como ingreso extra. Usa el celular todo el día pero no sabe nada de IA.

- **Problema:** saca las fotos de noche o en un cuarto mal iluminado; salen con ruido y movidas, y vende menos. Además suele tener un fondo poco prolijo (cama, piso).
- **Frustraciones:** las apps de "mejorar foto" cambian los colores o "arreglan" la prenda; le preocupa que el comprador reclame porque la prenda se ve mejor que en la realidad.
- **Objetivo:** fotos más limpias sin ocultar defectos reales (manchas, desgaste). Opcionalmente, un fondo más prolijo sin que el producto cambie.

## Declaración del proyecto

Mi sistema permite que Camila (vendedora de ropa usada) pueda limpiar el ruido y el desenfoque de las fotos de sus productos aplicando Stable Diffusion 1.5 con ControlNet tile a baja intensidad, y verificando con RT-DETR que no aparezcan ni desaparezcan objetos. Le advierte que los detalles finos fueron reconstruidos por una IA y le muestra un mapa de diferencias para que revise que no se tapó ningún defecto real. Como paso opcional, puede cambiar el fondo de la foto: ahí la IA solo genera el fondo y el producto conserva sus píxeles.

## Estructura

```
app.py               Interfaz Gradio (MVP: sub-tarea B)
src/imagenes.py      Carga y preparación de imágenes
src/diffusion.py     Difusión (SD 1.5 + ControlNet tile)
src/clasico.py       Filtros clásicos y corrección de iluminación (sin IA)
src/analysis.py      Análisis visual con RT-DETR
src/honesty.py       Mapa de diferencias, aviso e informes
src/metricas.py      PSNR, SSIM y degradación reproducible de fotos
src/segmentacion.py  Paso 2 opcional: máscara del producto con BiRefNet
src/escena.py        Paso 2 opcional: fondo con SD inpainting y composición
exploracion/         App completa de la exploración (solo fondo), fuera del MVP
scripts/             Pruebas sin interfaz y evaluación PSNR/SSIM
colab/               Notebooks para ejecutar en Google Colab o Kaggle
docs/                Registro de decisiones y evidencias
tests/               Verificaciones mínimas
```

## Instalación

```bash
git clone https://github.com/alvarito1709/imagenes-TI-marketplace-gen.git
cd imagenes-TI-marketplace-gen
uv sync
uv run app.py
```

Para ejecutarla con GPU, usar `colab/ejecutar_en_colab.ipynb` (Google Colab) o `colab/ejecutar_en_kaggle.ipynb` (Kaggle).

## Exploración fuera del MVP: separar producto y escena

La consigna pide elegir **una** sub-tarea (A, B o C). Elegimos la **B**. Antes de confirmarlo exploramos otra idea que no encaja en ninguna de las tres, y por eso quedó fuera del MVP, pero la dejamos documentada porque enseña algo sobre lo real y lo inventado.

**De dónde salió.** Al probar la sub-tarea B con fotos reales, el modelo **alteraba logos, botones y estampados del producto** y, a intensidad baja, casi no mejoraba nada. Además, en las fotos reales de marketplace que reunió Álvaro, el problema grande era el fondo mal tomado y la mala luz. Probamos entonces **separar producto y escena**: BiRefNet recorta el producto, Stable Diffusion inpainting genera solo el fondo y los píxeles originales del producto se pegan encima.

| Original | Retocado por la IA (sub-tarea B) | Mapa de diferencias |
|---|---|---|
| ![Original](docs/evidencias/primer%20modelado/bolso_original1.png) | ![Retocado](docs/evidencias/primer%20modelado/bolso_retocado.png) | ![Mapa de calor](docs/evidencias/primer%20modelado/mapa_calor_bolso.webp) |

**Qué pasó.** Con el producto a la vista, el inpainting inventó objetos (una persona que llevaba el bolso, un soporte de estudio). Al generar el fondo **sin que el modelo vea el producto**, en un cuadrado de 512 px, el resultado mejoró mucho: el producto queda intacto, pero con límites que se ven en la captura (la máscara incluyó un dedo de la mano que sostenía el bolso, y no hay sombra de contacto).

![Fondo cambiado](docs/evidencias/remover%20fondo/bolso_fondo_cambiado.webp)

**Por qué quedó fuera del MVP.** Cambiar el fondo no es denoising, super-resolución ni restauración de fotos antiguas, y la consigna exige una de esas tres. Por eso el núcleo es la sub-tarea B, y el fondo se integró solo como **paso 2 opcional** de la app (se carga la primera vez que se usa). El detalle de la decisión y las alternativas está en [`docs/decisiones.md`](docs/decisiones.md). El estado exacto de la exploración está en la etiqueta de git `exploracion-escena` y el código en `src/segmentacion.py`, `src/escena.py` y `exploracion/`.

## Pendiente de completar

Qué es real y qué es inventado · Arquitectura · Modelos y licencias · Hardware de prueba · Decisiones de diseño (HAI) · Conceptos de procesamiento digital · Uso de agentes de IA · Limitaciones · Capturas y video demo.
