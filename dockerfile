FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the PATH
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY . .

# Install dependencies using uv
RUN uv pip install --system \
    fastapi \
    pydantic \
    uvicorn \
    aiofiles \
    pytesseract \
    numpy \
    sentence_transformers \
    markdown \
    scikit-learn \
    python-dateutil \
    requests \
    httpx \
    faker \
    pillow

# Expose port for FastAPI
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]