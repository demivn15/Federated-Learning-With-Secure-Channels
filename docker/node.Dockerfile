# Imagen de un nodo FL (CN1–CN5 e infra) para las pruebas en contenedores (E9, E10) antes de las Pi.
# Solo lleva las dependencias: el código del proyecto se monta en /app en solo lectura al lanzar el
# contenedor (experiments/docker_launcher.py), así el hash de código del manifiesto es el del PC y no hay
# que reconstruir la imagen tras cada cambio.
#   docker build -f docker/node.Dockerfile -t fl-node:dev docker
FROM python:3.11.9-slim-bookworm

# iproute2: `tc netem` para inyectar latencia y pérdida (E9); requiere --cap-add NET_ADMIN
RUN apt-get update \
 && apt-get install -y --no-install-recommends iproute2 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-node.txt /tmp/requirements-node.txt
RUN pip install --no-cache-dir -r /tmp/requirements-node.txt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app
WORKDIR /app
ENTRYPOINT ["python"]
