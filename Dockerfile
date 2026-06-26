FROM python:3.11-slim

WORKDIR /app

COPY requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

EXPOSE 10000

CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-10000}"]
