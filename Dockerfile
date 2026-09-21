FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer between builds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project.
COPY data/ data/
COPY src/ src/
COPY api/ api/

# Train the model at build time so the image is self-contained and starts
# instantly at runtime. (For a larger dataset you'd instead train once and
# COPY a pre-built models/ directory in, to keep image builds fast.)
RUN python src/train.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
