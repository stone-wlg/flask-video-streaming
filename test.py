import glob
from os.path import basename, splitext
from base64 import b64decode, b64encode

# location = './images'

# fileset = [file for file in glob.glob(location + "**/*.jpg", recursive=False)]

# for file in fileset:
#     print(splitext(basename(file))[0])

# image_handle = open("1.jpg", "rb")
# raw_image_data = image_handle.read()
# encoded_data = b64encode(raw_image_data)

# print(raw_image_data)
# print(encoded_data)

#data:image/jpg;base64,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


import datetime, time

faces = []
def check_faces(name):
    has_face = False
    tmp_faces = faces.copy()
    for face in tmp_faces:
      if (datetime.datetime.now() - face["timeout"]).total_seconds() >= 3:
          print("imhere 1")
          faces.remove(face)

      if face["name"] == name:
          print("imhere 2")
          has_face = True
          break
      
    if not has_face:
      faces.append({"name": name, "timeout": datetime.datetime.now()})

check_faces("stone")
time.sleep(1)
check_faces("stone")
time.sleep(1)
check_faces("stone")
time.sleep(1)
check_faces("stone")
print(faces)
