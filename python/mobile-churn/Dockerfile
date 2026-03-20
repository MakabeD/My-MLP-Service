FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "fastapi[standard]"
COPY configs/ /app/configs/
COPY artifacts/models/ /app/artifacts/models/
COPY artifacts/preprocess/ /app/artifacts/preprocess/
COPY src/ /app/src/
COPY utils/ /app/utils
EXPOSE 8000
CMD ["fastapi", "run", "src/server/main.py", "--port", "8000"]