# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Installar dependences
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of files 
COPY . .

# Streamlit port
EXPOSE 8501

# Comand for running app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]