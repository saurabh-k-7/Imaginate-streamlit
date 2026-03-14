# Use a lightweight Python base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for OpenCV and segmentify)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    mesa-libGL \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 1: Install heavy AI libraries first (to utilize cache)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Step 2: Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 3: Copy the app code
COPY . .

# Step 4: Expose Streamlit port
EXPOSE 8501

# Step 5: Run the app with explicit server settings
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
