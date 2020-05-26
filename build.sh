#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

docker build -t stonewlg/flask-video-streaming:latest .
