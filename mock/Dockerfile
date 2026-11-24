FROM python:3.12-slim
WORKDIR /mock
RUN pip install --no-cache-dir fastapi==0.115.6 "uvicorn[standard]==0.34.0" python-multipart==0.0.20
COPY server.py .
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
