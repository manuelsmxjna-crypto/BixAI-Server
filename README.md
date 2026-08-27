# BixAI Server

Backend reproducible de BixStudio para quitar fondos y ampliar imágenes x4 con ONNX Runtime. Esta versión conserva el contrato del servicio `0.1.0` y añade validación Cloudflare Turnstile antes de consumir GPU.

## Endpoints

- `GET /health` — estado de Turnstile, modelos y proveedores ONNX.
- `POST /remove-background?alpha_mode=multiply`
- `POST /upscale?alpha_mode=binary`

Los POST usan `multipart/form-data` y requieren:

- `image`: PNG, JPEG o WEBP real.
- `cf-turnstile-response`: token fresco emitido para la acción correspondiente.

Las acciones válidas son `remove_background` y `upscale`. Cloudflare hace los tokens de un solo uso; una repetición se rechaza en `siteverify`.

## Variables de entorno

- `TURNSTILE_SECRET` — obligatoria y secreta. Debe configurarse en Secret Manager/Cloud Run, nunca en Git.
- `TURNSTILE_ALLOWED_HOSTNAMES` — hostnames exactos autorizados para emitir tokens de Turnstile.
- `ALLOWED_ORIGINS` — orígenes HTTPS exactos autorizados por CORS para llamar al backend.
- `MAX_UPLOAD_MB` — por defecto `30`.
- `MAX_PIXELS` — por defecto `50000000`.
- `BG_MODEL_PATH`, `UP_MODEL_PATH`, `BG_MODEL_SIZE`, `UP_TILE`, `UP_PAD`, `UP_SCALE` — ajustes de modelos.

## Modelos

Copia antes del build:

```powershell
Copy-Item ..\Builder-current\models\bg-remove.onnx .\models\
Copy-Item ..\Builder-current\models\realesr-anime-x4.onnx .\models\
```

Los `.onnx` están ignorados por Git para evitar subir más de 100 MB accidentalmente.

## Pruebas

```bash
python -m pip install -r requirements-test.txt
python -m pytest
```

## Seguridad

La validación es del lado servidor contra el endpoint canónico de Cloudflare. Además se comprueban acción, hostname, bytes, píxeles y formato real decodificado. El servidor falla cerrado si `TURNSTILE_SECRET` no está configurado.

El despliegue no forma parte de esta reconstrucción inicial: primero debe probarse en una revisión separada para no reemplazar por accidente el servicio GPU que ya funciona.

`cloudbuild.yaml` genera la etiqueta `turnstile-f27fccf`. `.gcloudignore` incluye los modelos locales en el contexto de construcción, mientras `.gitignore` evita subirlos a GitHub.
