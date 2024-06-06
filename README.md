# PJA-ASI-12C-GR1-API

## Description

This is an api for PJA-ASI-12C-GR1 project to show models.

## Endpoints
http://127.0.0.1:8000/predict - for showing the model data

## Docker commands

BUILD: docker build -t pja-asi-12c-gr1-api .

START: docker run -d -p 8000:8000 --name pja-asi-12c-gr1-api -e PORT=8000 pja-asi-12c-gr1-api
