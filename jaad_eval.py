# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import cv2
import numpy as np
from os import listdir
from os.path import basename,isfile, join
import scipy.sparse
import scipy.io as sio
from sklearn import metrics
import matplotlib.pyplot as plt


# Evaluation Methods
def parse_rec(filename, occlusion,hrng):
	""" Parse a jaad file """

	bbs = []
	with open(filename) as f:
		annot_lines = [x.strip() for x in f.readlines()]
	for l in annot_lines:
		elements = l.split(' ')
		# check for pedestrian class tag, occlusion tag and height
		if elements[0].find('ped') != -1:
			if not occlusion and int(elements[5]):
				continue
		else:
			if int(elements[4]) > hrng[0] and int(elements[4]) < hrng[1]:
				bbs.append(elements)

  # Load object bounding boxes into a data frame.
	objects= []
	for ix, bbox in enumerate(bbs):
		obj_struct = {}
		obj_struct['name'] = 'pedestrian'
		obj_struct['pose'] = ''
		obj_struct['truncated'] = 0
		obj_struct['difficult'] = 0
		# Make pixel indexes 0-based
		obj_struct['bbox'] = [int(bbox[1]), #- 1
							  int(bbox[2]), #- 1
							  int(bbox[1]) + int(bbox[3]), #- 1
							  int(bbox[2]) + int(bbox[4])] #- 1]

		objects.append(obj_struct)

	return objects
def voc_ap(rec, prec):
	"""
	Compute VOC AP given precision and recall.
	"""

	# correct AP calculation
	# first append sentinel values at the end
	mrec = np.concatenate(([0.], rec, [1.]))
	mpre = np.concatenate(([0.], prec, [0.]))

	# compute the precision envelope
	for i in range(mpre.size - 1, 0, -1):
		mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

	# to calculate area under PR curve, look for points
	# where X axis (recall) changes value
	i = np.where(mrec[1:] != mrec[:-1])[0]

	# and sum (\Delta recall) * prec
	ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
	return ap
def jaad_eval(detpath,
			 annopath,
			 imagesetfile,
			 classname,
			 cachedir,
			 occlusion = False,
			 hrng = [50, float("Inf")],
			 ovthresh=0.5,
			 use_07_metric=False):
  # first load gt
	if not os.path.isdir(cachedir):
		os.mkdir(cachedir)
	cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)

	imagenames = [f.split('.')[0] for f in \
	sorted(listdir(imagesetfile))\
	if isfile(os.path.join(imagesetfile, f))]
	if not os.path.isfile(cachefile):
	# load annotations
		recs = {}
		for i, imagename in enumerate(imagenames):
			recs[imagename] = parse_rec(annopath.format(imagename),occlusion,hrng)
			if i % 100 == 0:
				print('Reading annotation for {:d}/{:d}'.format(
				i + 1, len(imagenames)))
	# save
		print('Saving cached annotations to {:s}'.format(cachefile))
		with open(cachefile, 'w') as f:
			pickle.dump(recs, f)
	else:
	# load
		with open(cachefile, 'rb') as f:
			try:
				recs = pickle.load(f)
			except:
				recs = pickle.load(f, encoding='bytes')

				# extract gt objects for this class
	class_recs = {}
	npos = 0
	for imagename in imagenames:
		R = [obj for obj in recs[imagename] if obj['name'] == classname]
		bbox = np.array([x['bbox'] for x in R])
		difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
		det = [False] * len(R)
		npos = npos + sum(~difficult)
		class_recs[imagename] = {'bbox': bbox,
							 'difficult': difficult,
							 'det': det}

							 # read dets
	detfile = detpath.format(classname)
	with open(detfile, 'r') as f:
		lines = f.readlines()

	splitlines = [x.strip().split(' ') for x in lines]
	image_ids = [x[0] for x in splitlines]
	confidence = np.array([float(x[1]) for x in splitlines])
	BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

	nd = len(image_ids)
	tp = np.zeros(nd)
	fp = np.zeros(nd)

	if BB.shape[0] > 0:
		# sort by confidence
		sorted_ind = np.argsort(-confidence)
		sorted_scores = np.sort(-confidence)
		BB = BB[sorted_ind, :]
		image_ids = [image_ids[x] for x in sorted_ind]

		# go down dets and mark TPs and FPs
		for d in range(nd):
			R = class_recs[image_ids[d]]
			bb = BB[d, :].astype(float)
			ovmax = -np.inf
			BBGT = R['bbox'].astype(float)

			if BBGT.size > 0:
				# compute overlaps
				# intersection
				ixmin = np.maximum(BBGT[:, 0], bb[0])
				iymin = np.maximum(BBGT[:, 1], bb[1])
				ixmax = np.minimum(BBGT[:, 2], bb[2])
				iymax = np.minimum(BBGT[:, 3], bb[3])
				iw = np.maximum(ixmax - ixmin + 1., 0.)
				ih = np.maximum(iymax - iymin + 1., 0.)
				inters = iw * ih

				# union
				uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
					   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
					   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

				overlaps = inters / uni
				ovmax = np.max(overlaps)
				jmax = np.argmax(overlaps)

			if ovmax > ovthresh:
				if not R['difficult'][jmax]:
					if not R['det'][jmax]:
						tp[d] = 1.
						R['det'][jmax] = 1
					else:
						fp[d] = 1.
			else:
				fp[d] = 1.

	# compute precision recall
	fp = np.cumsum(fp)
	tp = np.cumsum(tp)
	rec = tp / float(npos)
	# avoid divide by zero in case the first detection matches a difficult
	# ground truth
	prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
	ap = voc_ap(rec, prec, use_07_metric)

	return rec, prec, ap
# Visualization Methods
def visualize_detections(im, class_name, dets, thresh=0.5):
	"""Draw detected bounding boxes."""
	inds = np.where(dets[:, -1] >= thresh)[0]
	if len(inds) == 0:
		return
	im = im[:, :, (2, 1, 0)]
	fig, ax = plt.subplots(figsize=(12, 12))
	ax.imshow(im, aspect='equal')
	for i in inds:
		bbox = dets[i, :4]
		score = dets[i, -1]

		ax.add_patch(
			plt.Rectangle((bbox[0], bbox[1]),
						  bbox[2] - bbox[0],
						  bbox[3] - bbox[1], fill=False,
						  edgecolor='red', linewidth=3.5)
			)
		ax.text(bbox[0], bbox[1] - 2,
				'{:s} {:.3f}'.format(class_name, score),
				bbox=dict(facecolor='blue', alpha=0.5),
				fontsize=14, color='white')

	ax.set_title(('{} detections with '
				  'p({} | box) >= {:.1f}').format(class_name, class_name,
												  thresh),
				  fontsize=14)
	plt.axis('off')
	plt.tight_layout()
	plt.draw()
def visualize_classification(im_path, predictions,
bbox, class_name, thresh=0.5, show = 'scene', mode = 'exclusive'):

	# show: scene, shows classification in the full scene
	#		image, just the classified image
	# exclusive: displays one claas over the other
	im = cv2.imread(im_path)
	im = im[:, :, (2, 1, 0)] # rearrange channels from cv2 to rgb

	fig,ax = plt.subplots()
	if show is 'image':
		print(im.shape)
		print(bbox)
		im = im[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
		print(im.shape)

	ax.imshow(im)# aspect='equal'
	classes = [cl  for i, cl in enumerate(class_name)
					if predictions[i] >= thresh]
	if show is 'image':
		ax.set_title(classes,fontsize=14)
	else:
		ax.add_patch(
				plt.Rectangle((bbox[0], bbox[1]),
							bbox[2] - bbox[0],
							bbox[3] - bbox[1], fill=False,
							edgecolor='red', linewidth=3.5))

		ax.text(bbox[0], bbox[1] - 2,
					classes,
					bbox=dict(facecolor='blue', alpha=0.5),
					fontsize=14, color='white')

	plt.axis('off')
	plt.tight_layout()
	plt.draw()
	plt.show(fig)
def visualize_classification_cv(im_path, predictions,
bbox, class_name, thresh=0.5, show = 'scene', mode = 'exclusive'):

	# show: scene, shows classification in the full scene
	#		image, just the classified image
	# exclusive: displays one claas over the other
	im = cv2.imread(im_path)

	classes = [cl  for i, cl in enumerate(class_name)
					if predictions[i] >= thresh]
	bbox = np.array(bbox)
	if show is 'image':
		im = im[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
		s = 450/np.array(im.shape[0])
		im = cv2.resize(im,(0,0), fx = s, fy = s)
		for i,cl in enumerate(classes):
			cv2.putText(im,
			cl, # text '{}'.format(classes)
			(10,(i+1)*20), #origin
			cv2.FONT_HERSHEY_SIMPLEX, # font type
			0.7, # fontScale
			(0, 255, 0)) # color
	else:
		cv2.rectangle(im,(bbox[0], bbox[1]),
						 (bbox[2] , bbox[3]),
						 (255,0,0),1)

		cv2.putText(im,
			'{}'.format(classes), # text '{}'.format(classes)
			(bbox[0], bbox[1]), #origin
			cv2.FONT_HERSHEY_SIMPLEX, # font type
			0.7, # fontScale
			(0, 255, 0)) # color

	cv2.imshow("Output", im)
	cv2.waitKey(0)

def visualize_trajectory_predict_cv(im_path, actual , pred = None, scale = 0.5, window_name = 'trajectory'):

	for i in range(0,len(im_path)):
		im = cv2.imread(im_path[i])
		cv2.rectangle(im,(int(actual[i][0]) , int(actual[i][1])),
						 (int(actual[i][2]) , int(actual[i][3])),
						 (0,255,0) , 2)

		if pred is not None:
			cv2.rectangle(im,(int(pred[i][0]) , int(pred[i][1])),
							 (int(pred[i][2]) , int(pred[i][3])),
							 (255,0,0) , 2)

		im = cv2.resize(im,(0,0), fx = scale, fy = scale)

		cv2.imshow(window_name, im)
		cv2.waitKey(0)

def jaad_evaluate_classification(pred, gts, classes):

	# Precision: tp/(tp+fp) = tp/n
	# Recall = tp/(tp+fn)
	# average_preision: Area under curve of precision vs Recall
	# label_ranking: 1/n(sigma_n(1/|y_i|(sigma_y_ij=1(|rank(y_Ij=1)/rank_ij))))
	# ROC : auc of TPR vs FPR
	# AP = 1/11(sigma_recall) Precision(recall)
	# Miss rate = (FP+FN)/total
	# True Positive Rate (TPR) = TP/Actual yes
	# FRP = FP/actual no
	# prevelance = Actual yes /total

	acc_per_class = [metrics.average_precision_score(gts[:,i], pred[:,i])
	 for i in range(pred.shape[1])]
	print('Accuracy per class for {}: {}'.format(classes,acc_per_class))

	label_ranking = metrics.label_ranking_average_precision_score(
	gts, pred)
	print('Label ranking score:{}'.format(label_ranking))
def save_classification_results(pred, save_result_path):
	with open(save_result_path, 'wt') as fid:
		for p in pred:
			fid.write(' '.join(['%.3f' % i for i in p]))
			fid.write('\n')
def parse_detection_pedestrian(filename):
	with open(filename) as f:
		annot_lines = [x.strip() for x in f.readlines()]
	objects = []
	for l in annot_lines:
		elements = l.split(' ')
		obj_struct = {}
		obj_struct['image_path'] = elements[0]
		# Make pixel indexes 0-based
		obj_struct['bbox'] = [float(elements[1]), #- 1
							  float(elements[2]), #- 1
							  float(elements[3]),
							  float(elements[4])]
		obj_struct['score'] = float(elements[5])
		objects.append(obj_struct)
	return objects
def parse_detection_pedestrian_roc(filename):
	with open(filename) as f:
		annot_lines = [x.strip() for x in f.readlines()]
	objects = {}
	for l in annot_lines:
		elements = l.split(' ')
		if elements[0] not in objects.keys():
			objects[elements[0]] = {'bbox':[[float(elements[1]), #- 1
								  float(elements[2]), #- 1
								  float(elements[3]),
								  float(elements[4])]],
								  'score':[float(elements[5])]}
		else:
			objects[elements[0]]['bbox'].append([float(elements[1]), #- 1
												float(elements[2]), #- 1
												float(elements[3]),
												float(elements[4])])
			objects[elements[0]]['score'].append(float(elements[5]))
	return objects
def comp_overlaps( gt_boxes, det_boxes, ignore):
	num_dets = len(det_boxes)
	num_gts = len(gt_boxes)
	overlap_area = np.zeros((num_dets,num_gts))

	dets_width = det_boxes[:,2] - det_boxes[:,0]
	dets_height = det_boxes[:,3] - det_boxes[:,1]
	dets_area = dets_width * dets_height

	gts_width = gt_boxes[:,2] - gt_boxes[:,0]
	gts_height = gt_boxes[:,3] - gt_boxes[:,1]
	gts_area = gts_width * gts_height

	for i in range(num_dets):
		for j in range(num_gts):
			width = np.minimum(det_boxes[i,2],gt_boxes[j,2]) - np.maximum(det_boxes[i,0],gt_boxes[j,0])
			if width <= 0:
				continue
			height = np.minimum(det_boxes[i,3],gt_boxes[j,3]) - np.maximum(det_boxes[i,1],gt_boxes[j,1])
			if height <= 0:
				continue
			intersection = width * height
			if ignore[j]:
				union = dets_area[i]
			else:
				union = dets_area[i] + gts_area[j] - intersection
			overlap_area[i,j]= intersection/union
	return overlap_area
def evaluate_results(gts, dets, ovthresh = 0.5):

	dets = np.array([(*b,s,0) for b,s in zip(dets['bbox'],dets['score'])])
	gts = np.array([(*b,d) for b,d in zip(gts['boxes'],gts['difficult'])]).astype(float)

	if not gts.size:
		gts = np.zeros((0,5))
	if not dets.size:
		dets = np.zeros((0,6))

	num_dets = len(dets)
	num_gts = len(gts)
	dets_sorted_ind = np.argsort(-dets[:,4])
	dets = dets[dets_sorted_ind,:]
	gts_sorted_ind = np.argsort(gts[:,4])
	gts = gts[gts_sorted_ind,:]
	gts[:,4] = -gts[:,4]
	overlap_area = comp_overlaps(gts[:,0:4], dets[:,0:4], gts[:,4] == -1)
	for d in range(num_dets):
		best_overlap = ovthresh
		best_gt = 0
		best_match = 0
		for g in range(num_gts):
			match = gts[g,4]
			if match == 1:
				continue
			if best_match != 0 and match == -1:
				break
			if overlap_area[d,g] < best_overlap:
				continue
			best_overlap = overlap_area[d,g]
			best_gt = g
			if match == 0:
				best_match = 1
			else:
				best_match = -1
		g = best_gt
		match = best_match
		if match == -1:
			dets[d,5] = match
		elif match == 1:
			gts[g,4] = match
			dets[d,5] = match

	return gts,dets
def jaad_evaluate_detection_map(detpath,
			 imdb, ovthresh=0.5):

	# extract gt objects for this class
	annotations = imdb.gt_annotation_evaluation()
	npos = 0
	class_recs = {}
	for key, value in annotations.items():
		det = [False] * len(annotations[key]['boxes'])
		difficult = np.array(annotations[key]['difficult']).astype(np.bool)
		npos = npos + sum(~difficult)
		class_recs[key] = {'bbox':annotations[key]['boxes'],
							'difficult': difficult,
							'det':det}
	# read dets
	detections = parse_detection_pedestrian(detpath)

	image_ids = [x['image_path'] for x in detections]
	confidence = np.array([x['score'] for x in detections])
	BB = np.array([x['bbox'] for x in detections])

	nd = len(image_ids)
	tp = np.zeros(nd)
	fp = np.zeros(nd)

	if BB.shape[0] > 0:
		# sort by confidence
		sorted_ind = np.argsort(-confidence)
		sorted_scores = np.sort(-confidence)
		BB = BB[sorted_ind, :]
		image_ids = [image_ids[x] for x in sorted_ind]

		# go down dets and mark TPs and FPs
		for d in range(nd):
			R = class_recs[image_ids[d]]
			bb = BB[d, :].astype(float)
			ovmax = -np.inf
			BBGT = R['bbox'].astype(float)

			if BBGT.size > 0:
				# compute overlaps
				# intersection
				ixmin = np.maximum(BBGT[:, 0], bb[0])
				iymin = np.maximum(BBGT[:, 1], bb[1])
				ixmax = np.minimum(BBGT[:, 2], bb[2])
				iymax = np.minimum(BBGT[:, 3], bb[3])
				iw = np.maximum(ixmax - ixmin + 1., 0.)
				ih = np.maximum(iymax - iymin + 1., 0.)
				inters = iw * ih

				# union
				uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
					   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
					   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

				overlaps = inters / uni
				ovmax = np.max(overlaps)
				jmax = np.argmax(overlaps)

			if ovmax > ovthresh:
				if not R['difficult'][jmax]:
					if not R['det'][jmax]:
						tp[d] = 1.
						R['det'][jmax] = 1
					else:
						fp[d] = 1.
			else:
				fp[d] = 1.

	# compute precision recall
	fp = np.cumsum(fp)
	tp = np.cumsum(tp)
	rec = tp / float(npos)
	# avoid divide by zero in case the first detection matches a difficult
	# ground truth
	prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
	ap = voc_ap(rec, prec)
	print('mAP: %.3f' % ap)
	return rec, prec, ap
def jaad_evaluate_pedestrian_detection_roc(detpath, imdb, ovthresh=0.5, ref_start = -2):
	# read dets
	annotations = imdb.gt_annotation_evaluation()
	detections = parse_detection_pedestrian_roc(detpath)
	gts = {}
	dets = {}
	# value in annotations.items():
	keys = sorted(annotations.keys())
	for key in keys:
		if key not in detections.keys():
			detect = {'bbox': [], 'score':[]}
		else:
			detect = detections[key]
		gts[key], dets[key] = evaluate_results(annotations[key],detect)

	num_images = len(gts)
	assert(len(dets) == num_images) , \
	'The number of detections and ground truth annotations does not match'

	gts = np.concatenate([gts[key] for key in sorted(gts.keys())], axis = 0)
	dets = np.concatenate([dets[key] for key in sorted(dets.keys())], axis = 0)

	idx_gt_difficult = np.where(gts[:,4]!= -1)
	gts = gts[idx_gt_difficult[0],:]

	idx_det_difficult = np.where(dets[:,5]!=-1)
	dets = dets[idx_det_difficult[0],:]

	roc_ref = np.power(10,np.arange(ref_start, 0.25, 0.25))
	m = len(roc_ref)
	num_positives = len(gts)
	score = dets[:,4]
	tp = dets[:,5]


	order = np.argsort(-score)
	score = score[order]
	tp = tp[order]
	fp = np.array(tp[:] !=1).astype(np.float64)

	fp = np.cumsum(fp)
	tp = np.cumsum(tp)


	xs = fp/num_images
	ys = tp/num_positives
	xs1 = np.insert(xs,0,-np.inf)
	ys1 = np.insert(ys,1,0)
	for i in range(m):
		j = np.where(xs1 <= roc_ref[i])[0]
		roc_ref[i] = ys1[j[-1]]

	# print(np.log(np.maximum(1e-10,1-roc_ref)))
	miss_rate = np.exp(np.mean(np.log(np.maximum(1e-10,1-roc_ref))));
	print('Miss rate (MR%.f): %.3f' % (-ref_start, miss_rate))
	return miss_rate
def jaad_write_results_python(box, img_name, score, fid):
	fid.write('%s '%(img_name)) #idx+1
	fid.write(' '.join(['%.3f' % i for i in box]))
	fid.write(' %.3f' % score)
	fid.write('\n')
def jaad_write_results_matlab(box, idx, score, fid):
	box[2] = box[2] - box[0]
	box[3] = box[3] - box[1]
	fid.write('%s,'%(idx+1)) #
	fid.write(','.join(['%.3f' % i for i in box]))
	fid.write(',%.3f' % score)
	fid.write('\n')

def jaad_trajectory_mse(y_batch, o_batch):

	not_exist_pid = 0

	y = tf.reshape(y_batch, (-1, pxy_dim))
	o = tf.reshape(o_batch, (-1, out_dim))

	pids = y[:, 0]

	# remain only existing pedestrians data
	exist_rows = tf.not_equal(pids, not_exist_pid)
	y_exist = tf.boolean_mask(y, exist_rows)
	o_exist = tf.boolean_mask(o, exist_rows)
	pos_exist = y_exist[:, 1:]

	# compute 2D normal prob under output parameters
	log_prob_exist = normal2d_log_pdf(o_exist, pos_exist)
	# for numerical stability
	log_prob_exist = tf.minimum(log_prob_exist, 0.0)

	loss = -log_prob_exist
	return loss

	def compute_abe(x_true, x_pred):
		"""Compute Average displacement error (ade).
		In the original paper, ade is mean square error (mse) over all estimated
		points of a trajectory and the true points.
		:param x_true: (n_samples, seq_len, max_n_peds, 3)
		:param x_pred: (n_samples, seq_len, max_n_peds, 3)
		:return: Average displacement error
		"""
		# pid != 0 means there is the person at the frame.
		not_exist_pid = 0
		exist_elements = x_true[..., 0] != not_exist_pid

		# extract pedestrians positions (x, y), then compute difference
		pos_true = x_true[..., 1:]
		pos_pred = x_pred[..., 1:]
		diff = pos_true - pos_pred

		# ade = average displacement error
		ade = np.mean(np.square(diff[exist_elements]))
		return ade
