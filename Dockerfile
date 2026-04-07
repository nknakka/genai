FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8080

CMD ["adk", "api_server", "--host", "0.0.0.0", "--port", "8080"]
