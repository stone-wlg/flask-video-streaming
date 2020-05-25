import glob
from os.path import basename, splitext

location = './images'

fileset = [file for file in glob.glob(location + "**/*.jpg", recursive=False)]

for file in fileset:
    print(splitext(basename(file))[0])
