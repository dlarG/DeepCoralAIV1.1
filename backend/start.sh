#!/bin/bash
python -m pip install --upgrade pip
pip install -r requirements.txt
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 app:app