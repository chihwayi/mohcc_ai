FROM python:3.12

# Install required system libraries for Python dependencies
RUN apt-get update && apt-get install -y \
    libbz2-dev \
    libncurses5-dev \
    libffi-dev \
    libreadline-dev \
    libssl-dev \
    zlib1g-dev

# Set the working directory inside the container
WORKDIR /app

# Copy the contents of the local directory to the container
COPY . /app

# Install Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
