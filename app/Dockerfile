FROM python:3.11-slim-buster

RUN apt-get update && apt-get install libglib2.0-0 libgl1-mesa-glx  -y

WORKDIR /app
COPY . .

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r  /tmp/requirements.txt

EXPOSE 5000

ENV MODEL_PATH="/app/models/trained_model.keras"

CMD ["python", "app.py"]
