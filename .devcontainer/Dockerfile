FROM mcr.microsoft.com/devcontainers/python:3.11

WORKDIR /workspaces/${localWorkspaceFolderBasename}

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install marimo && \
    pip install -r requirements.txt
