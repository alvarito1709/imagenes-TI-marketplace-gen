# Exploración fuera del MVP: separar producto y escena

El MVP de este trabajo es la **sub-tarea B** (denoising/deblurring con difusión): ver `app.py` en la raíz.

Esta carpeta guarda la app de una exploración que hicimos antes de volver a B: segmentar el producto
con BiRefNet, generar **solo el fondo** con inpainting y pegar encima los píxeles originales del producto.
Se hizo porque, en las fotos reales de marketplace que reunió Álvaro, el problema grande era el fondo mal
tomado y la mala iluminación, y porque el modelo de la sub-tarea B alteraba logos y estampados.

- Por qué se hizo y por qué no es el MVP: `docs/decisiones.md`.
- Estado exacto de la exploración: etiqueta de git `exploracion-escena`.
- Código: `src/segmentacion.py`, `src/escena.py`, `scripts/prueba_escena.py` y los tests asociados.
- Ejecutar (necesita GPU, ver `colab/`): `python exploracion/app_escena.py`
