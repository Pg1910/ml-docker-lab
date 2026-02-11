SHELL := /bin/bash

TRAINER_IMAGE := ml-docker-lab:0.3
NETWORK := ml-docker-lab_default
MLFLOW_URI := http://mlflow:5000
EXP := california-housing-rf-v2
ARTIFACT_VOL := ml_artifacts

SEED := 42
N_EST := 300
MAX_DEPTH := 20

.PHONY: up down ps logs build train

up:
	docker compose up -d --build
	docker compose ps

down:
	docker compose down

ps:
	docker compose ps

logs:
	docker compose logs --no-color --tail 120 mlflow

build:
	docker build -t $(TRAINER_IMAGE) .

train:
	docker run --rm \
	  --network $(NETWORK) \
	  -e MLFLOW_TRACKING_URI=$(MLFLOW_URI) \
	  -e MLFLOW_EXPERIMENT_NAME=$(EXP) \
	  -v $(ARTIFACT_VOL):/artifacts \
	  $(TRAINER_IMAGE) \
	  --seed $(SEED) --n-estimators $(N_EST) --max-depth $(MAX_DEPTH)
