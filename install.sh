#!/bin/bash

mkdir models
cd models

wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
wget https://pjreddie.com/media/files/yolov3-tiny.weights
wget https://raw.githubusercontent.com/opencv/opencv/next/data/haarcascades/haarcascade_frontalface_default.xml
wget https://raw.githubusercontent.com/opencv/opencv/next/data/haarcascades/haarcascade_fullbody.xml
wget https://raw.githubusercontent.com/opencv/opencv/next/data/lbpcascades/lbpcascade_frontalface_improved.xml
wget https://raw.githubusercontent.com/opencv/opencv/next/data/lbpcascades/lbpcascade_profileface.xml

cd ..
pip install -r requirements.txt --break-system-packages
