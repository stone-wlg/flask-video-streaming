import os
import cv2
from base_camera import BaseCamera
import face_recognition
import numpy as np
import glob
import time
from os.path import basename, splitext
from base64 import b64decode, b64encode

class Camera(BaseCamera):
    video_source = "0"

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(os.environ['OPENCV_CAMERA_SOURCE'])
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def get_image_encode_string(filename):
        with open(filename, "rb") as image: 
            return b64encode(image.read())       

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            print('Could not start camera.')
            return

        known_face_encodings = []
        known_face_names = []
        filefullnames = [filefullname for filefullname in glob.glob("./images/*.jpg", recursive=False)]
        for filefullname in filefullnames:
            image = face_recognition.load_image_file(filefullname)
            face_encoding = face_recognition.face_encodings(image)[0] 
            known_face_encodings.append(face_encoding)
            known_face_names.append(splitext(basename(filefullname))[0])

        print(known_face_names)

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        while True:
            # Grab a single frame of video
            ret, frame = camera.read()
            if not ret:
                print("Could not read camera.")
                time.sleep(3)
                yield "data:image/jpg;base64,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"

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()
