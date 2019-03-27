#!/usr/bin/env bash

case $(whoami) in
    "tomek") PY=/Users/tomek/.virtualenvs/masters/bin/python ;;
    "paperspace") PY=python3 ;;
esac

STRATEGY="undersampling"
if [[ ! -z $1 ]]; then
    STRATEGY=$1
fi

echo "Using '$STRATEGY' strategy"

cd src/
${PY} main.py --submit --strategy=${STRATEGY}


