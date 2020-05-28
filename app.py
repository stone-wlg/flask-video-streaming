#!/usr/bin/env python
from importlib import import_module
import os
import dao
from flask import Flask, render_template, Response, jsonify

# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera

# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/top/<int:limit>')
def top(limit):
    data = dao.query("""
        SELECT json_build_object(
            'id', u.id,
            'name', u.name,
            'pinyin', u.pinyin,
            'department', u.department,
            'image', h.image,
            'ts', h.ts      
            )::jsonb
        FROM public.user u
            INNER JOIN public.history h
            ON h.pinyin = u.pinyin 
        ORDER BY h.ts DESC
        LIMIT %s;     
    """ % limit)
    return jsonify(data)

@app.route('/total/<int:limit>')
def total(limit):
    data = dao.query("""
        SELECT json_build_object(
            'ts', htc.bucket, 
            'total_cnt', htc.cnt,
            'total_invalid_cnt', COALESCE(hti.cnt, 0),
            'total_valid_cnt', (htc.cnt - COALESCE(hti.cnt, 0))
            )::jsonb
        FROM history_total_count_in_5mins htc
        LEFT JOIN history_total_invalid_count_in_5mins hti
        ON hti.bucket = htc.bucket
        ORDER BY htc.bucket DESC
        LIMIT %s;   
    """ % limit)
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True) #, ssl_context=('./certs/server.pem', './certs/server.key')
