#! /bin/bash

export HANDWRITING_GENERATION_DATA_DIR=$PWD/data
export HANDWRITING_GENERATION_MODEL_DIR=$PWD/models

pip install -r requirements.txt
python3 setup.py install

cd notebooks

jupyter notebook
