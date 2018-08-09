#USAGE: python face-alignment.py
# original code: https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
# db_name: dataset folder name
# generate new aligned data in 'oulu_align'


from imutils import face_utils
from imutils.face_utils import FaceAligner
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
import csv
import pickle

full_filename = '/data/OriginalImg/VL/Strong/P012/Anger/000.jpeg'
full_filename2 = '/data/OriginalImg/VL/Strong/P012/Anger/017.jpeg'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('preprocessing/shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)

image = cv2.imread(full_filename)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 2)
print(rects)

# loop over the face detections
for (i, rect) in enumerate(rects):
  faceAligned = fa.align(image, gray, rect)
  faceAligned = faceAligned[35:256-31, 33:256-33]
  cv2.imwrite('aligned2.png', faceAligned)

  # cv2.imshow('aligned2', faceAligned)

image = cv2.imread(full_filename)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)
print(rects)

# loop over the face detections
for (i, rect) in enumerate(rects):
  faceAligned2 = fa.align(image, gray, rect)
  faceAligned2 = faceAligned2[35:256-31, 33:256-33]
  cv2.imwrite('aligned1.png', faceAligned2)
  # cv2.waitKey(0)
  # cv2.imwrite(full_filename.replace(db_name, new_name), faceAligned)
