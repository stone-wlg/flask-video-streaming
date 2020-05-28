import glob
import math
import os
import os.path
import pickle
import threading
import time
from base64 import b64decode, b64encode
from os.path import basename, splitext

import cv2
import face_recognition
import numpy as np
from face_recognition.face_recognition_cli import image_files_in_folder
from PIL import Image, ImageDraw
from sklearn import neighbors
import datetime
import dao

from base_camera import BaseCamera


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
    def frames():
        # print("Training KNN classifier...")
        # classifier = train("images/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
        # print("Training complete!")
        # process one frame in every 30 frames for speed
        process_this_frame = 59
        print('Setting cameras up...')
        cap = cv2.VideoCapture(Camera.video_source)
        while 1 > 0:
            if not cap.isOpened():
                cap = cv2.VideoCapture(Camera.video_source)
                print('Could not open camera.')
                time.sleep(3)
                return        

            ret, frame = cap.read()
            if ret:
                # Different resizing options can be chosen based on desired program runtime.
                # Image resizing for more stable streaming
                img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                process_this_frame = process_this_frame + 1
                if process_this_frame % 60 == 0:
                    predictions = predict(img, model_path="trained_knn_model.clf")
                frame = show_prediction_labels_on_image(frame, predictions)
                saved_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                check_valid_faces(predictions, saved_frame)
                check_invalid_faces(predictions, saved_frame)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()


valid_faces = []
def check_valid_faces(predictions, frame):
    global valid_faces

    tmp_valid_faces = valid_faces.copy()
    for face in tmp_valid_faces:
        if (datetime.datetime.now() - face["timeout"]).total_seconds() >= 10:
            print("remove valid face: " + face["name"])
            valid_faces.remove(face)
            # todo
            # mqtt send name
            dao.insert("INSERT INTO public.history (pinyin, image) VALUES ('" + face["name"] + "', '/images/" + face["name"] + ".jpg')")
            # cv2.imwrite("./images/valid/" + face["name"] + ".jpg", face["frame"])

    for name, _ in predictions:
        if name == "unknown":
            return 

        has_valid_face = False
        tmp_valid_faces = valid_faces.copy()
        for face in tmp_valid_faces:
            if face["name"] == name:
                has_valid_face = True
                break
        
        if not has_valid_face and len(valid_faces) < 100:
            valid_faces.append({"name": name, "timeout": datetime.datetime.now()})

    print(valid_faces)

invalid_faces = []
invalid_faces_timeout = datetime.datetime.now()
def check_invalid_faces(predictions, frame):
    global invalid_faces
    global invalid_faces_timeout

    tmp_invalid_faces = invalid_faces.copy()
    for face in tmp_invalid_faces:
        if (datetime.datetime.now() - face["timeout"]).total_seconds() >= 10:
            print("remove invalid face: " + face["name"])
            invalid_faces.remove(face)
            # todo
            dao.insert("INSERT INTO public.history (pinyin, image) VALUES ('unknown', '/images/invalid/" + face["name"] + ".jpg')")
            cv2.imwrite("./images/invalid/" + face["name"] + ".jpg", face["frame"])

    if (datetime.datetime.now() - invalid_faces_timeout).total_seconds() <= 3:
        print("invalid_faces_timeout return")
        return

    for name, _ in predictions:
        if name != "unknown":
            return 
        
        invalid_faces_timeout = datetime.datetime.now()

        if len(invalid_faces) < 10:
            invalid_username = "unknown-" + str(datetime.datetime.now().timestamp())
            invalid_faces.append({"name": invalid_username, "frame": frame, "timeout": datetime.datetime.now()})      

    print(invalid_faces)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_frame: frame to do the prediction on.
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_face_locations = face_recognition.face_locations(X_frame)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(frame, predictions):
    """
    Shows the face recognition results visually.

    :param frame: frame to show the predictions on
    :param predictions: results of the predict function
    :return opencv suited image to be fitting with cv2.imshow fucntion:
    """
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # enlarge the predictions for the full sized image.
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    # Save image in open-cv format to be able to show it.

    opencvimage = np.array(pil_image)
    return opencvimage


def interval_train():
    while True:
        print(datetime.datetime.now().strftime("%Y-%b-%d %H:%M:%S") + ": Training KNN classifier...")
        classifier = train("images/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
        print(datetime.datetime.now().strftime("%Y-%b-%d %H:%M:%S") + ": Training complete!")
        time.sleep(30)   

th2 = threading.Thread(target=interval_train)
th2.setDaemon(True) 
th2.start()