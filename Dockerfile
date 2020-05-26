FROM stonewlg/face_recognition:latest

COPY . /flask-video-streaming

RUN cd /flask-video-streaming && pip install -r ./requirements.txt

WORKDIR /flask-video-streaming

EXPOSE 5000

ENTRYPOINT [ "/flask-video-streaming/start.sh" ]
