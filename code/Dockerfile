# Use the official Python 3.11 image from Docker Hub
FROM python:3.11-slim

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
# Make sure to include gunicorn in your requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y gcc && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# The port number the container should expose
# Use the environment variable PORT if it's set, otherwise default to 8000
EXPOSE $PORT

# Run the application
CMD gunicorn --workers=4 --bind 0.0.0.0:${PORT:-8000} app:app
