# Container with both the Streamlit UI (port 8501) and the API (port 8000).
#   docker build -t docqa-capstone .
#   docker run -p 8501:8501 -p 8000:8000 --env-file .env docqa-capstone
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/app/storage

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501 8000

# Start the API in the background and the UI in the foreground.
CMD uvicorn src.api:app --host 0.0.0.0 --port 8000 & \
    streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
