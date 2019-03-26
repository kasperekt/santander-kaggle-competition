#!/usr/bin/env bash

if [[ ! -d "out/" ]]; then
    mkdir out
fi

if [[ ! -d "data/" ]]; then
    mkdir data
    kaggle competitions download -c santander-customer-transaction-prediction -p data/
    unzip 'data/*.zip' -d data/
    chmod ugo+wr data/*.csv
fi