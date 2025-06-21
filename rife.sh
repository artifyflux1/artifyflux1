#!/usr/bin/bash

git clone https://github.com/megvii-research/ECCV2022-RIFE rife
cd rife

python3.10 -m venv venv
curl -kL https://bootstrap.pypa.io/get-pip.py | venv/bin/python

venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m pip install sk-video moviepy opencv-python

wget https://huggingface.co/aka7774/ECCV2022-RIFE/resolve/main/RIFE_trained_model_v3.6.zip
unzip -o RIFE_trained_model_v3.6.zip
