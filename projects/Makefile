# Makefile for tenfold experiments

TRAIN_CONFIG ?= projects/configs/tenfold_train.yaml
PRED_CONFIG ?= projects/configs/tenfold_pred.yaml

train:
	python -m projects.apps.tenfold_train_appv3 --config $(TRAIN_CONFIG)

pred:
	python -m projects.apps.tenfold_pred_app --config $(PRED_CONFIG)

all: train pred

.PHONY: train pred all

# make -f projects/Makefile all
# make -f projects/Makefile train