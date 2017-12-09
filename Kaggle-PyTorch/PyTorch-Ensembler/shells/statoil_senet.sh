#!/usr/bin/env bash
dataset=statoil
epochs=57

python ../../main.py --dataset ${dataset} --epochs ${epochs}
