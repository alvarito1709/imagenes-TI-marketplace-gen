# Instrucciones para agentes de IA (Claude Code, Codex, Copilot, etc.)

## Proyecto
Trabajo Integrador N°2 — Procesamiento Digital de Imágenes (IFTS 24, 2026).
App que limpia ruido y desenfoque de fotos de productos con difusión (sub-tarea B de la consigna)
para una vendedora de marketplace (persona: Camila), **sin ocultar defectos reales** y avisando qué
fue reconstruido por la IA. Paso opcional posterior: cambiar el fondo conservando el producto.
Autor: Álvaro Ortiz. Responsabilidad del código: de él.

## Stack
- Python 3.12, gestor `uv`. Interfaz: Gradio.
- Difusión (MVP): SD 1.5 + ControlNet tile (`diffusers`). Análisis: RT-DETR (`transformers`).
- Paso opcional de fondo: BiRefNet (segmentación, MIT) + SD 1.5 inpainting. Estado en `docs/decisiones.md`.
- Imágenes: Pillow, NumPy, OpenCV. Se ejecuta en Google Colab o Kaggle (la PC de desarrollo no tiene GPU).

## Comandos
- Instalar: `uv sync`
- Tests: `uv run pytest`
- App: `uv run app.py`

## Reglas
- **Regla central: lo que la IA cambia nunca se oculta.** Toda foto procesada sale con mapa de diferencias e informe en lenguaje simple.
- En el paso opcional de fondo, los píxeles del producto no se regeneran: la IA solo genera el fondo.
- La sub-tarea elegida es **B (denoising/deblurring)**; no cambiar de sub-tarea sin confirmarlo con Álvaro (la consigna exige una de las tres).
- Un paso chico por vez; mostrar el plan antes de escribir código.
- Commits chicos y descriptivos (un cambio, un commit). Ramas: `feature/*` → `develop` → `main`.
- Modelos se cargan UNA sola vez, nunca dentro de la función que corre en cada clic.
- Mensajes para la persona usuaria: español rioplatense simple, sin jerga, con qué hacer ante un error.
- Nunca subir tokens ni claves. No agregar funcionalidades fuera del alcance (3 funciones core).
- Cada decisión relevante se anota en `docs/decisiones.md` indicando si la tomó Álvaro o la propuso el agente.
