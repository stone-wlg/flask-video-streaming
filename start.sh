#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

export CAMERA="opencv"
export OPENCV_CAMERA_SOURCE="http://209.206.162.229/mjpg/video.mjpg"

python ./app.py
