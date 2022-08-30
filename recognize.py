"""
@author: Astitva Prakash
Created on: 10-10-2020 20:08
"""
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

def recog(args):
	print("[INFO] Loading Face Detector...")
	protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
	modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
	print("[INFO] Loading Face Recognizer...")
	embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
	recognizer = pickle.loads(open(args["recognizer"], "rb").read())
	le = pickle.loads(open(args["le"], "rb").read())

	image = cv2.imread(args["image"])
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]
	imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB = False, crop = False)
	detector.setInput(imageBlob)
	detections = detector.forward()

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = image[startY:endY, startX: endX]
			(fH, fW) = face.shape[:2]
			if fW < 20 or fH < 20:
				continue
			faceBlob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			prob = preds[j]
			name = le.classes_[j]

			text = "{}: {:.2f}%".format(name, prob * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
			cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	cv2.imshow("Image", image)
	cv2.waitKey(0)
	
if __name__ == '__main__':
	arg = argparse.ArgumentParser()
	arg.add_argument("-i", "--image", required=True, help="Path to inout images")
	arg.add_argument("-d", "--detector", required=True, help="Path OpenCV DL Detector")
	arg.add_argument("-m", "--embedding-model", required=True, help="Path to OpenCV Embedding Model")
	arg.add_argument("-r", "--recognizer", required=True, help="Path to Trained Model")
	arg.add_argument("-l", "--le", required=True, help="Path to Label Encoder")
	arg.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum confidence to filter detections")
	args = vars(arg.parse_args())
	recog(args)