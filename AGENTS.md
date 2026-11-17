# Instrucciones para agentes de IA (Claude Code, Codex, Copilot, etc.)

## Proyecto
Trabajo Integrador N°2 — Procesamiento Digital de Imágenes (IFTS 24, 2026).
App que mejora el fondo y la iluminación de fotos de productos para una vendedora de
marketplace (persona: Camila), **sin alterar el producto ni ocultar defectos reales**
y avisando qué fue generado por la IA (el fondo). Autor: Álvaro Ortiz. Responsabilidad del código: de él.

## Stack
- Python 3.12, gestor `uv`. Interfaz: Gradio.
- Segmentación: BiRefNet (`transformers`, MIT). Fondo: SD 1.5 inpainting (`diffusers`). Análisis: RT-DETR (`transformers`).
- `src/diffusion.py` (SD 1.5 + ControlNet tile) es el enfoque anterior, plan B hasta el 18/11 a las 10:00.
- Imágenes: Pillow, NumPy, OpenCV. Se ejecuta en Google Colab o Kaggle (la PC de desarrollo no tiene GPU).

## Comandos
- Instalar: `uv sync`
- Tests: `uv run pytest`
- App: `uv run app.py`

## Reglas
- **Regla central: los píxeles del producto nunca se regeneran.** La IA solo toca el fondo; el producto original se pega encima al final.
- Un paso chico por vez; mostrar el plan antes de escribir código.
- Commits chicos y descriptivos (un cambio, un commit). Ramas: `feature/*` → `develop` → `main`.
- Modelos se cargan UNA sola vez, nunca dentro de la función que corre en cada clic.
- Mensajes para la persona usuaria: español rioplatense simple, sin jerga, con qué hacer ante un error.
- Nunca subir tokens ni claves. No agregar funcionalidades fuera del alcance (3 funciones core).
- Cada decisión relevante se anota en `docs/decisiones.md` indicando si la tomó Álvaro o la propuso el agente.
