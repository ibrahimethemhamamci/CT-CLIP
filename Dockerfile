
FROM python:3.11-slim AS base_image

WORKDIR /app

RUN pip install --upgrade pip setuptools
RUN apt-get update && \
    apt-get install -y \
      gcc \
      git \
      libhdf5-serial-dev \
      libgl1 \
      libgl1-mesa-glx \
      libglib2.0-0 \
      pkg-config \
    && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip3 install .

#
#FROM python:3.11-slim AS download_model
#WORKDIR /app
#COPY download.py .
#RUN pip install torch transformers huggingface_hub
#ENV TRUST_REMOTE_CODE=true
#RUN --mount=type=secret,id=HUGGINGFACE_TOKEN \
#  HUGGINGFACE_TOKEN=$(cat /run/secrets/HUGGINGFACE_TOKEN) \
#  ./download.py


FROM base_image AS app

WORKDIR /app
#COPY --from=download_model /app/models /app/models
#COPY CT_CLIP_zeroshot.pt /app/models/
COPY src /app/src
RUN pip3 install .
COPY app.py /app

EXPOSE 8000
ENTRYPOINT ["/bin/sh", "-c"]
CMD ["uvicorn app:app --host 0.0.0.0 --port 8000"]
