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
| 2026-11-17 | **Cambio de enfoque:** de retocar el producto (ControlNet tile) a separar producto y escena (BiRefNet + SD inpainting solo en el fondo) | **Decidió Álvaro** | Motivos, alternativas y evidencia en la sección «Cambio de enfoque» de más abajo |
| 2026-11-17 | La decisión «Fondo mal tomado queda fuera de alcance» queda **reemplazada** por el cambio de enfoque | Álvaro | Se conserva la fila anterior como historial |
| 2026-11-17 | Segmentación con BiRefNet (licencia MIT) y no con RMBG-2.0 | Propuso Claude (verificado en las fichas de los modelos); aprobó Álvaro | RMBG-2.0 es de uso no comercial y exige aceptar condiciones para acceder |
| 2026-11-17 | Fondos disponibles: blanco de estudio, madera clara, gris suave (generados con IA) y blanco liso sin IA como comparación | Propuso Claude; aprobó Álvaro | El blanco liso sin IA sirve de línea base clásica |
| 2026-11-17 | `diffusion.py` (enfoque 1) se conserva como **plan B** hasta el miércoles 18/11 a las 10:00 | Álvaro | A esa hora se decide seguir con el enfoque nuevo o volver al anterior |

## Cambio de enfoque (17/11/2026): de retocar el producto a mejorar la escena

**Qué se hizo primero (enfoque 1).** Stable Diffusion 1.5 + ControlNet tile en modo img2img, para limpiar ruido y desenfoque de la foto del producto. La evidencia está en `docs/evidencias/primer modelado/` (originales, resultados, mapas de diferencias y un video de la prueba).

**Qué se observó al probarlo en Kaggle con fotos reales:**
1. El modelo **cambiaba el producto en sí**: logos, botones y estampados salían alterados. Es exactamente lo que el sistema tenía que evitar, porque Camila necesita que el comprador vea el producto tal como es.
2. A intensidad baja, en cambio, **casi no mejoraba la imagen**: la mala iluminación y el fondo mal tomado seguían igual.

**Por qué Álvaro decidió cambiar de rumbo.** Por dos motivos:
- **El modelo contradecía el propósito del sistema.** Reconstruir detalles del producto lo vuelve menos fiel, y con modelos de difusión los detalles chicos con significado (texto, logos, botones, patrones) son los que peor se conservan.
- **Aportaba más valor a la persona usuaria.** Al mirar fotos reales de marketplace, a Álvaro le pareció que lo que más le sirve a una vendedora es mejorar la presentación (fondo e iluminación), no el ruido o el desenfoque del producto.

**Alternativas evaluadas:**

| Alternativa | Resultado |
|---|---|
| Seguir con el retoque a baja intensidad | Descartada como camino principal: no resuelve fondo ni luz. Se conserva como plan B |
| Edición por instrucción (FLUX.1 Kontext, Qwen-Image-Edit) | Descartada: modelos pesados con licencias y memoria exigentes, y reescriben el producto con la misma libertad, así que se esperaba el mismo problema de logos |
| Cambiar la postura del producto | Descartada: obliga a re-sintetizar el producto entero y multiplica lo inventado |
| **Separar producto y escena (elegida)** | El producto conserva sus píxeles originales y la IA solo genera el fondo |

**Diseño elegido:**

```
foto → preparar → [corregir luz, opcional]
     → BiRefNet: máscara del producto → refinar (erosión + suavizado de borde)
     → SD inpainting regenera SOLO el fondo (máscara invertida)
     → componer: los píxeles ORIGINALES del producto van encima del fondo nuevo
     → verificación + informe
```

- **Qué es real y qué es inventado, ahora:** el producto es real, el fondo es generado y la luz es un ajuste de brillo y contraste.
- **Cómo se verifica:** porcentaje de píxeles del producto que cambiaron (0% sin corrección de luz), mapa de diferencias (producto sin cambios, fondo cambiado) y RT-DETR antes y después, con su limitación de siempre: solo conoce las 80 clases de COCO y no reconoce ropa.
- **Modelos:** BiRefNet (MIT) para segmentar y `stable-diffusion-v1-5/stable-diffusion-inpainting` (CreativeML OpenRAIL-M) para generar el fondo.

**Qué no cambia:** la persona (Camila), el principio de honestidad hacia el comprador, RT-DETR como análisis visual y la ejecución en GPU remota.

**Riesgos conocidos al decidir:**
- Una mala máscara sobre ropa en fondos parecidos (cama, piso) puede recortar mal el producto.
- El producto puede quedar "flotando" si el modelo no genera sombra de contacto, y los bordes pueden dejar un halo.
- No está verificado cuántas dependencias extra necesita BiRefNet ni cuánto tarda cada foto en la GPU de Kaggle.
- La cuota de GPU gratuita es impredecible.

**Plan B y punto de decisión.** `src/diffusion.py` y la interfaz del enfoque 1 siguen en el repositorio. El miércoles 18/11 a las 10:00 se decide: si el fondo generado sale aceptable con las fotos de prueba, se sigue con el enfoque nuevo; si no, se vuelve al enfoque 1 y el enfoque nuevo se documenta como trabajo futuro.
