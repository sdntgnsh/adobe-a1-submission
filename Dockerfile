# Use the official Python 3.10 image on linux/amd64 architecture
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install the Python dependencies
# This happens during the 'docker build' step
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files into the container
COPY . .

# Set the command to run when the container starts
# This will execute 'python main.py'
CMD ["python", "main.py"]