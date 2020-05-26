#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

export CAMERA=${CAMERA:-"opencv"}
export OPENCV_CAMERA_SOURCE=${OPENCV_CAMERA_SOURCE:-"http://192.168.1.100:8080/?action=stream?dummy=param.mjpg"}

python ./app.py
