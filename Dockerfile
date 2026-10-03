FROM python:3.10-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

ENV DDON_MODEL_PATH="/app/model log/20260520_09-39_AM+model.Epoch-27_Loss-0.000404+MSE-0.000244+Batsize-[512].h5"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
