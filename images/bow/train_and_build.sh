#!/bin/bash
python train_bow.py
docker build . --tag=bow:test
