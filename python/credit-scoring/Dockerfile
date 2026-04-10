FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "fastapi[standard]"
COPY config/ /app/config/
COPY models/ /app/models/
COPY preprocess/ /app/preprocess/
COPY src/ /app/src/
EXPOSE 8000
CMD ["fastapi", "run", "src/server/main.py", "--port", "8000"]