# For more information, please refer to https://aka.ms/vscode-docker-python
FROM ubuntu:22.04
FROM python:3.8

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
WORKDIR /app
VOLUME /app_vol
COPY . /app
RUN python -m pip install -r requirements.txt
RUN apt-get update && apt-get install libgl1 -y

ENTRYPOINT  ["python", "main.py"]