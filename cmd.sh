#!/bin/bash

set -e
if [ "$ENV" = 'DEV' ]; then
  echo "Running development server"
  exec python3 /distllm/server.py
elif [ "$ENV" = 'COMPUTE_NODE' ]; then
  exec python3 -u deploy_node.py
elif [ "$ENV" = 'CLIENT' ]; then
  exec python3 -u manager.py
else
  echo "Running uwsgi server"
  exec uwsgi --http 0.0.0.0:9090 --wsgi-file /distllm/server.py \
     --callable app --stats 0.0.0.0:9191
fi
