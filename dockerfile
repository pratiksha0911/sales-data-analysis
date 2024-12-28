# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies for Streamlit and machine learning
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && apt-get clean

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "eapp.py", "--server.port=8501", "--server.enableCORS=false"]
