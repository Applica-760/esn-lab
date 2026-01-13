# Makefile for tenfold experiments

TRAIN_CONFIG ?= projects/configs/tenfold_train.yaml
EVAL_CONFIG ?= projects/configs/tenfold_eval.yaml

train:
	python -m projects.apps.tenfold_train_appv3 --config $(TRAIN_CONFIG)

eval:
	python -m projects.apps.tenfold_eval_app --config $(EVAL_CONFIG)

all: train eval

.PHONY: train eval all

# make -f projects/Makefile all
# make -f projects/Makefile train