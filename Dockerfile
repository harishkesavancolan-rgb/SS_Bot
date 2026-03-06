# ─────────────────────────────────────────────────────────────
# Dockerfile — RAG Ingestion Pipeline
# ─────────────────────────────────────────────────────────────
#
# What this file does, step by step:
#
#   1. Starts from an official Python image (like a clean laptop
#      with Python already installed)
#   2. Copies your code into the container
#   3. Installs your dependencies (requirements.txt)
#   4. Sets the command to run when the container starts
#
# To build:  docker build -t rag-ingest .
# To run:    docker run --env-file .env rag-ingest
# ─────────────────────────────────────────────────────────────


# ── Stage 1: Base image ───────────────────────────────────────
# We use the official slim Python 3.11 image.
# "slim" means it's a smaller version — no extras we don't need.
FROM python:3.11-slim


# ── Stage 2: Set working directory ───────────────────────────
# All commands from here on run inside /app inside the container.
# Think of it as: cd /app
WORKDIR /app


# ── Stage 3: Install system dependencies ─────────────────────
# pdfplumber needs these system libraries to read PDFs properly.
# We clean up apt cache at the end to keep the image size small.
RUN apt-get update && apt-get install -y \
    libpoppler-cpp-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*


# ── Stage 4: Install Python dependencies ─────────────────────
# We copy requirements.txt FIRST (before the rest of the code).
# Why? Docker caches each step. If requirements.txt hasn't changed,
# Docker skips reinstalling packages — making builds much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 5: Copy your code into the container ───────────────
# Now we copy everything else. This step re-runs only when
# your actual code changes.
COPY ingestion/ ./ingestion/


# ── Stage 6: Environment variables ───────────────────────────
# These are default values — safe, non-secret config.
# Secret values (AWS keys etc.) are passed in at runtime via .env
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1


# ── Stage 7: Default command ──────────────────────────────────
# When the container starts, run ingest.py.
# PDF path and OpenSearch host are passed as arguments:
#   docker run --env-file .env rag-ingest story.pdf my-host.es.amazonaws.com
CMD ["ingestion.ingest.handler"]
