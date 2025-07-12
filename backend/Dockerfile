FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libgl1-mesa-glx libgthread-2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app files
COPY . .

# Expose port and run
ENV PORT=8080
CMD ["python", "app.py"]
