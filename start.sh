#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

export CONNECTION=${CONNECTION:-"dbname=robot user=postgres password=robot@321 host=localhost port=5432"}
export CAMERA=${CAMERA:-"opencv_knn"}
export OPENCV_CAMERA_SOURCE=${OPENCV_CAMERA_SOURCE:-"http://10.125.23.221:8080/?action=stream?dummy=param.mjpg"}

python ./app.py
