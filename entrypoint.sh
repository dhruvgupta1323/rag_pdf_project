#!/bin/sh
set -e

# Default SERVICE_TYPE to backend if not specified
SERVICE_TYPE="${SERVICE_TYPE:-backend}"

echo "Starting RAG application with SERVICE_TYPE: $SERVICE_TYPE"

if [ "$SERVICE_TYPE" = "frontend" ]; then
    echo "Starting Streamlit frontend on port ${PORT:-8501}..."
    exec streamlit run app.py \
        --server.port="${PORT:-8501}" \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false

elif [ "$SERVICE_TYPE" = "both" ]; then
    echo "Starting FastAPI backend on internal port 8000..."
    uvicorn main:app --host 0.0.0.0 --port 8000 &
    
    echo "Starting Streamlit frontend on public port ${PORT:-8501}..."
    exec streamlit run app.py \
        --server.port="${PORT:-8501}" \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false

else
    echo "Starting FastAPI backend on port ${PORT:-8000}..."
    exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
fi
