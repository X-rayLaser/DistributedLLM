#!/bin/bash

set -e
if [ "$ENV" = 'DEV' ]; then
  echo "Running development server"
  exec python server.py
else
  echo "Running uwsgi server"
  exec uwsgi --http 0.0.0.0:9090 --wsgi-file /distllm/server.py \
     --callable app --stats 0.0.0.0:9191
fi