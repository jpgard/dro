"""
Pretrained open-source FaceNet implementation
github: https://github.com/jpgard/keras-facenet
h5 file in Google Drive: https://drive.google.com/drive/folders/12aMYASGCKvDdkygSv1yQq8ns03AStDO_
"""
from os import path as osp

from tensorflow.python.keras.saving.save import load_model


def load_facenet_model():
    cwd, _ = osp.split(__file__)
    facenet = load_model("facenet_keras.h5", compile=False)
    return facenet
