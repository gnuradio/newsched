#!/bin/sh

. ${PREFIX}/setup_env.sh 
cd ${PREFIX}/src/newsched/rpc/fastapi
uvicorn main:app --host 0.0.0.0 --port $REST_PORT --reload --factory