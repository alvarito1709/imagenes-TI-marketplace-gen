# Restaurador de fotos de productos

> Limpia el ruido y el desenfoque de las fotos de tus productos con IA, y te muestra qué cambió para que no se oculte ningún defecto real.

**Trabajo Integrador N°2** — Procesamiento Digital de Imágenes e Introducción a Visión por Computadora · IFTS 24 · 2026

> **Estado:** en desarrollo (esqueleto del proyecto).

## Persona usuaria

**Camila**, 28-35 años. Vende ropa y accesorios usados por marketplace como ingreso extra. Usa el celular todo el día pero no sabe nada de IA.

- **Problema:** saca las fotos de noche o en un cuarto mal iluminado; salen con ruido y movidas, y vende menos.
- **Frustraciones:** las apps de "mejorar foto" cambian los colores o "arreglan" la prenda; le preocupa que el comprador reclame porque la prenda se ve mejor que en la realidad.
- **Objetivo:** fotos más limpias sin ocultar defectos reales (manchas, desgaste).

## Declaración del proyecto

Mi sistema permite que Camila (vendedora de ropa usada) pueda limpiar el ruido y el desenfoque de las fotos de sus productos aplicando Stable Diffusion 1.5 con ControlNet tile a baja intensidad, y verificando con RT-DETR que no aparezcan ni desaparezcan objetos. Le advierte que los detalles finos fueron reconstruidos por una IA y le muestra un mapa de diferencias para que revise que no se tapó ningún defecto real.

## Estructura

```
app.py            Interfaz Gradio
src/imagenes.py   Carga y preparación de imágenes
src/diffusion.py  Difusión (SD 1.5 + ControlNet tile)
src/clasico.py    Línea base clásica sin IA
src/analysis.py   Análisis visual con RT-DETR
src/honesty.py    Mapa de diferencias, aviso e informe
colab/            Notebook para ejecutar en Google Colab
docs/             Registro de decisiones
tests/            Verificaciones mínimas
```

## Instalación

```bash
git clone https://github.com/alvarito1709/imagenes-TI-marketplace-gen.git
cd imagenes-TI-marketplace-gen
uv sync
uv run app.py
```

Para ejecutarla con GPU, usar `colab/ejecutar_en_colab.ipynb`.

## Pendiente de completar

Qué es real y qué es inventado · Arquitectura · Modelos y licencias · Hardware de prueba · Decisiones de diseño (HAI) · Conceptos de procesamiento digital · Uso de agentes de IA · Limitaciones · Capturas y video demo.
