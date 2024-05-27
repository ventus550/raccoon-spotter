FROM python:3.10-slim-buster

ARG MODEL_PATH=data/06_models/trained_model.keras

WORKDIR /
COPY app .
COPY ${MODEL_PATH} models/trained_model.keras
COPY src/raccoon_spotter/models/components imports/components
COPY src/raccoon_spotter/utils imports/utils

RUN apt-get update && apt-get install libglib2.0-0 libgl1-mesa-glx  -y
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]

