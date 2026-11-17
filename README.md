# Restaurador de fotos de productos

> Mejora el fondo y la iluminación de las fotos de tus productos con IA, sin tocar el producto, y te muestra qué cambió para que no se oculte ningún defecto real.

**Trabajo Integrador N°2** — Procesamiento Digital de Imágenes e Introducción a Visión por Computadora · IFTS 24 · 2026

> **Estado:** en desarrollo. El 17/11 se cambió el enfoque (ver [Cambio de enfoque](#cambio-de-enfoque)).

## Persona usuaria

**Camila**, 28-35 años. Vende ropa y accesorios usados por marketplace como ingreso extra. Usa el celular todo el día pero no sabe nada de IA.

- **Problema:** saca las fotos de noche o en un cuarto mal iluminado, con la cama o el piso de fondo; se ven oscuras y poco profesionales, y vende menos.
- **Frustraciones:** las apps de "mejorar foto" cambian los colores o "arreglan" la prenda; le preocupa que el comprador reclame porque la prenda se ve mejor que en la realidad.
- **Objetivo:** fotos con mejor fondo y luz, sin que el producto cambie ni se oculten defectos reales (manchas, desgaste).

## Declaración del proyecto

Mi sistema permite que Camila (vendedora de ropa usada) pueda mejorar el fondo y la iluminación de las fotos de sus productos, aplicando segmentación con BiRefNet y Stable Diffusion inpainting para generar **solo el fondo**, y verificando con un chequeo píxel a píxel y con RT-DETR que el producto no cambió. Le advierte que el fondo fue generado por una IA y que la corrección de luz modifica brillo y contraste.

## Estructura

```
app.py            Interfaz Gradio
src/imagenes.py   Carga y preparación de imágenes
src/diffusion.py  Enfoque 1 (SD 1.5 + ControlNet tile): plan B hasta el 18/11
src/clasico.py    Filtros clásicos y corrección de iluminación (sin IA)
src/segmentacion.py  (planificado) Máscara del producto con BiRefNet
src/escena.py     (planificado) Fondo generado con SD inpainting y composición
src/analysis.py   Análisis visual con RT-DETR
src/honesty.py    Mapa de diferencias, aviso e informe
colab/            Notebook para ejecutar en Google Colab
docs/             Registro de decisiones y evidencias
tests/            Verificaciones mínimas
```

## Instalación

```bash
git clone https://github.com/alvarito1709/imagenes-TI-marketplace-gen.git
cd imagenes-TI-marketplace-gen
uv sync
uv run app.py
```

Para ejecutarla con GPU, usar `colab/ejecutar_en_colab.ipynb` (Google Colab) o `colab/ejecutar_en_kaggle.ipynb` (Kaggle).

## Cambio de enfoque

El primer enfoque retocaba la foto con Stable Diffusion 1.5 + ControlNet tile. Al probarlo con fotos reales, el modelo **alteraba logos, botones y estampados del producto**, justo lo que el sistema debía evitar, y a baja intensidad no mejoraba la luz ni el fondo. Además, a Álvaro le pareció que mejorar la presentación (fondo e iluminación) aportaba más valor a una vendedora de marketplace que limpiar ruido. Por eso pasó a un enfoque que **separa producto y escena**: el producto conserva sus píxeles originales y la IA solo genera el fondo.

Evidencia del primer enfoque (el modelo alterando el producto):

| Original | Retocado por la IA | Mapa de diferencias |
|---|---|---|
| ![Original](docs/evidencias/primer%20modelado/bolso_original1.png) | ![Retocado](docs/evidencias/primer%20modelado/bolso_retocado.png) | ![Mapa de calor](docs/evidencias/primer%20modelado/mapa_calor_bolso.webp) |

El detalle de la decisión, las alternativas evaluadas y los riesgos están en [`docs/decisiones.md`](docs/decisiones.md).

## Pendiente de completar

Qué es real y qué es inventado · Arquitectura · Modelos y licencias · Hardware de prueba · Decisiones de diseño (HAI) · Conceptos de procesamiento digital · Uso de agentes de IA · Limitaciones · Capturas y video demo.
