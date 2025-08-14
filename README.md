# bbe sdk

SDK de evaluación black-box para modelos ML. Permite consumir batches de datos por medio de colas de rabbit y enviar las predicciones al servicio de calibración.

## Instalación

### A) Desde repo privado GitLab con Deploy Token
```bash
export GITLAB_USER="gitlab+deploy-token-XXXX"
export GITLAB_TOKEN="REEMPLAZAR_POR_TOKEN"
pip install "git+https://${GITLAB_USER}:${GITLAB_TOKEN}@gitlab.com/<grupo>/<bbe_sdk>.git@v0.1.0"
