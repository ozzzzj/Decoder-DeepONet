FROM python:3.10-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

ENV DDON_MODEL_PATH=/app/model/DDON.h5
ENV DDON_MODEL_URL=https://github.com/ozzzzj/Decoder-DeepONet/releases/download/DDON/20260520_09-39_AM%2Bmodel.Epoch-27_Loss-0.000404%2BMSE-0.000244%2BBatsize-.512.h5

RUN mkdir -p /app/model \
    && curl -L --fail --retry 3 "$DDON_MODEL_URL" -o "$DDON_MODEL_PATH"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
