"""
@author: Astitva Prakash
Created on: 11-10-2020 20:01
"""

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', default='dataset', help='Path to database to train on.')
ap.add_argument('-e', '--embeddings', default=os.path.sep.join(['output', 'embeddings.pickle']), help='Path to embeddings pickle.')
ap.add_argument('-f', '--detector', default='face_detection_model', help='Directory name containing OpenCV ResNet Model.')
ap.add_argument('-o', '--embedding-model', default='openface_nn4.small2.v1.t7', help='Path to OpenFace Neural Net')
ap.add_argument('-c', '--confidence', default=0.5, type=float, help='Minimum conofidence to filter results.')
ap.add_argument('-r', '--recognizer', default=os.path.sep.join(['output', 'recognizer.pickle']), help='Path to recognizer model pickle.')
ap.add_argument('-l', '--le', default=os.path.sep.join(['output', 'le.pickle']), help='Path to label encoder pickle.')
ap.add_argument('-i', '--image', help="Path to image if using recognizer within an image")
ap.add_argument('-v', '--video', type=int, help='Video Input ID to stream from. Use -v 0 to stream from integrated webcam.')
ap.add_argument('--all', action='store_true', help='Extract, train and, run the model. Suggested after updating the dataset.')
ap.add_argument('--extract', action='store_true', help='Only extract the embeddings from dataset. Suggested after updating the dataset.')
ap.add_argument('--train', action='store_true', help='Only train the model. Suggested after recalibration or updation of model on hyperparameters.')
ap.add_argument('--run', action='store_true', help='Only run the system. Suggested to, well, show off.')
args = ap.parse_args()

def decorator():
	print('------------------------------------')

args = vars(args)

if args['all'] or args['extract']:
	import extract_embeddings
	extract_embeddings.extract(args)
	print('Extractions Complete.......')
	decorator()

if args['all'] or args['train']:
	import train_model
	train_model.train(args)
	print('Completed Training.........')
	decorator()

if args['all'] or args['run']:
	if args['image'] == None and args['video'] == None:
		raise SystemExit('Need either image or video flag to run. Exiting...')
	if args['image'] != None:
		import recognize
		recognize.recog(args)
		print('Run Complete...........')
	if args['video'] != None:
		import recognize_video
		recognize_video.recog(args)
		print('Run Complete..........')