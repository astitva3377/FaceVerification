"""
@author: Astitva Prakash
Created on: 10-10-2020 17:22
"""

from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

def extract(args):
	print("[INFO] Loading facial detector...")
	protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
	modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
	print("[INFO] Reading face recognizer...")
	embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

	print("[INFO] Acquiring Faces...")
	imagePaths = list(paths.list_images(args["dataset"]))
	knownEmbeddings = []
	knownNames = []
	total = 0

	for (i, imagePath) in enumerate(imagePaths):
		print("[INFO] Processing image {}/{}".format(i+1, len(imagePaths)))
		name = imagePath.split(os.path.sep)[-2]
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width = 600)
		(h, w) = image.shape[:2]

		imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB = False, crop = False)
		detector.setInput(imageBlob)
		detections = detector.forward()

		if len(detections) > 0:
			i = np.argmax(detections[0, 0, :, 2])
			confidence = detections[0, 0, i, 2]
			if confidence > args["confidence"]:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				face = image[startY: endY, startX: endX]
				(fH, fW) = face.shape[:2]
				if fW < 20 or fH < 20:
					continue

				faceBlob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), swapRB = True, crop = False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				knownNames.append(name)
				knownEmbeddings.append(vec.flatten())
				total += 1

	print("[INFO] Serializing {} embeddings...".format(total))
	data = {"embeddings": knownEmbeddings, "names": knownNames}
	file = open(args["embeddings"], "wb")
	file.write(pickle.dumps(data))
	file.close()

if __name__ == '__main__':
	arg = argparse.ArgumentParser()
	arg.add_argument("-i", "--dataset", required=True, help="Path to input directory")
	arg.add_argument("-e", "--embeddings", required=True, help="Path to output facial embeddings DB")
	arg.add_argument("-d", "--detector", required=True, help="Path to OpenCV detector")
	arg.add_argument("-m", "--embedding-model", required=True, help="Path to OpenCV DL Face Model")
	arg.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum confidence to filter detections")
	args = vars(arg.parse_args())
	extract(args)