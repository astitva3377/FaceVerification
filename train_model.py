"""
@author: Astitva Prakash
Created on: 10-10-2020 18:57
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

def train(args):
	print("[INFO] Loading Facial Embeddings...")
	data = pickle.loads(open(args["embeddings"], "rb").read())
	print("[INFO] Encoding Labels...")
	le = LabelEncoder()
	labels = le.fit_transform(data["names"])

	print("[INFO] Training Model...")
	recognizer = SVC(C = 1.0, kernel="linear", probability=True)
	recognizer.fit(data["embeddings"], labels)

	file = open(args["recognizer"], "wb")
	file.write(pickle.dumps(recognizer))
	file.close()
	file = open(args["le"], "wb")
	file.write(pickle.dumps(le))
	file.close()

if __name__ == '__main__':
	arg = argparse.ArgumentParser()
	arg.add_argument("-e", "--embeddings", required=True, help="Path to Facial Embeddings DB")
	arg.add_argument("-r", "--recognizer", required=True, help="Path to Output Trained Model")
	arg.add_argument("-l", "--le", required=True, help="Path to Output Label Encoder")
	args = vars(arg.parse_args())
	train(args)