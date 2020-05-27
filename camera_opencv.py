import os
import cv2
from base_camera import BaseCamera
import face_recognition
import numpy as np
import glob
import time
import threading
from os.path import basename, splitext
from base64 import b64decode, b64encode

class Camera(BaseCamera):
    video_source = "0"

    known_face_encodings = []
    known_face_names = []

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(os.environ['OPENCV_CAMERA_SOURCE'])

        th2 = threading.Thread(target=Camera.reload_images)
        th2.setDaemon(True) 
        th2.start()

        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def get_image_encode_string(filename):
        with open(filename, "rb") as image: 
            return b64encode(image.read())       

    @staticmethod
    def reload_images():
        global known_face_encodings
        global known_face_names

        while (True):
            known_face_encodings_tmp = []
            known_face_names_tmp = []
            filefullnames = [filefullname for filefullname in glob.glob("./images/*.jpg", recursive=False)]
            for filefullname in filefullnames:
                image = face_recognition.load_image_file(filefullname)
                face_encoding = face_recognition.face_encodings(image)[0] 
                known_face_encodings_tmp.append(face_encoding)
                known_face_names_tmp.append(splitext(basename(filefullname))[0])
            
            Camera.known_face_encodings = known_face_encodings_tmp.copy()
            Camera.known_face_names = known_face_names_tmp.copy()

            print(Camera.known_face_names)  
            time.sleep(10)      

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            print('Could not start camera.')
            return

        known_face_encodings_tmp = []
        known_face_names_tmp = []
        filefullnames = [filefullname for filefullname in glob.glob("./images/*.jpg", recursive=False)]
        for filefullname in filefullnames:
            image = face_recognition.load_image_file(filefullname)
            face_encoding = face_recognition.face_encodings(image)[0] 
            known_face_encodings_tmp.append(face_encoding)
            known_face_names_tmp.append(splitext(basename(filefullname))[0])
        
        Camera.known_face_encodings = known_face_encodings_tmp.copy()
        Camera.known_face_names = known_face_names_tmp.copy()        
        print(Camera.known_face_names)

        # Initialize some variables
        process_this_frame = True

        while True:
            # Grab a single frame of video
            ret, frame = camera.read()
            if not ret:
                print("Could not read camera.")
                time.sleep(3)
                continue

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
                    matches = face_recognition.compare_faces(Camera.known_face_encodings, face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(Camera.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = Camera.known_face_names[best_match_index]

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
