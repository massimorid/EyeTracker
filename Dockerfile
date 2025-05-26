#select the python from the dockerhub with the version of the interpreter
FROM python:3.12

WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# No need to expose port since this is not a web application
# EXPOSE 5000

CMD ["python", "main.py"]