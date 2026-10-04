FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# pdfminer.six is pure python, but a couple of transitive deps need build tools.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY storage ./storage

EXPOSE 8000

# --app-dir matches WORKDIR; uvicorn needs to import the `app` package.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]