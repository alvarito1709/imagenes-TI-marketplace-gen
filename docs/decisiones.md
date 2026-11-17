# Registro de decisiones y uso de agentes de IA

Sirve para completar la sección "Uso de Agentes de IA" del README.

| Fecha | Decisión | Quién | Notas |
|---|---|---|---|
| 2026-11-16 | Sub-tarea B (denoising/deblurring) con SD 1.5 + ControlNet tile | eligió Álvaro | Alternativas descartadas: super-resolución, fotos antiguas |
| 2026-11-16 | Caso de uso: fotos de productos para vendedora de marketplace | eligió Álvaro | Ángulo: no ocultar defectos al comprador |
| 2026-11-16 | Persona: Camila | Propuso Claude, aprobó Álvaro | |
| 2026-11-16 | Ejecución en Google Colab | Álvaro | Su PC solo tiene GPU integrada (128 MB) |
| 2026-11-16 | Estructura del repo | Propuso Claude, aprobó Álvaro | Claude generó el esqueleto; Álvaro lo revisa antes de commitear |
| 2026-11-16 | `diffusion.py`: pipeline img2img + ControlNet tile, UniPC, 30 pasos, semilla fija 0 | Propuso Claude (basado en la ficha del modelo); Álvaro debe revisar | Semilla fija = resultados reproducibles |
| 2026-11-16 | Prompt neutro ("best quality, sharp, natural lighting") sin describir el producto | Propuso Claude | Para no inducir al modelo a inventar contenido |
| 2026-11-16 | Filtro de seguridad de SD desactivado | Propuso Claude; Álvaro debe confirmar | Evita falsos positivos (imagen negra) en fotos de ropa; el uso es solo restaurar fotos propias |
| 2026-11-16 | `restaurar` devuelve el mismo tamaño que la entrada | Propuso Claude | Necesario para el mapa de diferencias píxel a píxel |
| 2026-11-17 | Prueba en Kaggle: con strength 0.3 el modelo cambia muy poco y deja la mala luz y el fondo | Álvaro observó; Claude diagnosticó | ControlNet tile ancla la imagen al original: limpia ruido/desenfoque, no corrige luz ni fondo |
| 2026-11-17 | Corrección de iluminación clásica previa (gamma automático + CLAHE sobre L*a*b*) con casilla en la app | Propuso Claude; Álvaro debe validar con sus fotos | No inventa contenido; usa conceptos de la materia (histogramas, espacios de color) |
| 2026-11-17 | Fondo mal tomado queda fuera de alcance; se documenta como limitación | Propuso Claude | Cambiar el fondo exige segmentar y generar contenido nuevo: choca con la honestidad hacia el comprador |
