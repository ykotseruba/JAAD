from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import subprocess
import uuid
import cv2
import random
from os import listdir
import numpy as np
import time
import PIL
import keras

import xml.etree.ElementTree as ET
import scipy.sparse
import scipy.io as sio
import matplotlib.pyplot as plt
from .jaad_eval import *
from os.path import basename,isfile, join, abspath,isdir
from operator import add
from keras.preprocessing import image as image_keras
from sklearn.preprocessing import MultiLabelBinarizer
# from profilehooks import profile

# TODO: add support for different classes besides pedestrians

class jaad(object):
	def __init__(self, annotation_type = 'part',\
	use_occlusion = False, height_rng = [10 ,float('inf')],
	 squarify_ratio = 0, regen_pkl = False, data_path = '',
	  frame_skip = 10):

		self._year = '2016'
		self._name = 'jaad'
		self._image_set = 'train'
		self._regen_pkl = regen_pkl
		self._jaad_path = data_path if data_path else self._get_default_path()
		self._classes = ('__background__','pedestrian')
	# '__background__',  # always index 0
		self._num_classes = len(self._classes)
		self._class_to_ind = dict(list(zip(self._classes,
		list(range(self._num_classes)))))
		self._image_ext = '.png'
		#Object detection parameters
		self._annotation_type = annotation_type
		self._use_occlusion = use_occlusion
		self._squarify_ratio = squarify_ratio
		self._height_rng =  height_rng
		self._frame_skip = frame_skip
		self._video_ids = []
		# Default to roidb handler
		self._roidb_handler = self.gt_roidb
		self._roidb = None
		assert os.path.exists(self._jaad_path), \
		'Jaad path does not exist: {}'.format(self._jaad_path)
		# the seed for random selection of data points
		self._random_seed = 42
# Basic utilities
	@property
	def cache_path(self):
		cache_path = abspath(join(self._jaad_path, 'data_cache'))
		if not os.path.exists(cache_path):
			os.makedirs(cache_path)
		return cache_path
	def _get_default_path(self):
		"""
		Return the default path where jaad is expected to be installed.
		"""
		# return os.path.join('/home/aras/phd_work/project_files/data/' 'jaad')
		return os.path.join('/media/aras/Storage/datasets/data/', 'jaad')

# Data processing helpers
	@property
	def num_images(self):
		return len(self._image_index)
	def image_path_at(self, i):
		"""
		Return the absolute path to image i in the image sequence.
		"""
		return self.image_path_from_index(self._image_index[i])
	def image_path_from_index(self, index):
		"""
		Construct an image path from the image's "index" identifier.
		"""
		image_path = os.path.join(self._jaad_path,self._image_set, 'images', index + self._image_ext)
		assert os.path.exists(image_path), \
		'Path does not exist: {}'.format(image_path)
		return image_path
	def _load_image_set_index(self, frame_skip):
		"""
		Load the indexes listed in this dataset's image set file.
		"""
		image_set_folder = os.path.join(self._jaad_path,self._image_set, 'images')
		assert os.path.exists(image_set_folder), \
		'Path does not exist: {}'.format(image_set_folder)
		video_folders = [f for f in sorted(listdir(image_set_folder))
		if isdir(os.path.join(image_set_folder, f))]
		image_index = []
		for vfolder in video_folders:
			image_index.extend([vfolder +"/"+ f.split('.')[0] for idx, f in
			enumerate(sorted(listdir(os.path.join(image_set_folder,vfolder))))
			if isfile(os.path.join(image_set_folder,vfolder, f)) and
			(idx + 1) % frame_skip == 0])
		return image_index
	def _load_image_set_index_from_video_id(self, video_id, frame_skip):
		"""
		Load the indexes listed in this dataset's image set file.
		"""
		image_set_folder = os.path.join(self._jaad_path, self._image_set, 'images')
		assert os.path.exists(image_set_folder), \
		'Path does not exist: {}'.format(image_set_folder)
		image_index =[video_id +"/"+ f.split('.')[0] for idx, f in
			enumerate(sorted(listdir(os.path.join(image_set_folder,video_id))))
			if isfile(os.path.join(image_set_folder,video_id, f)) and
			(idx + 1) % frame_skip == 0]
		return image_index
	def image_name_from_index(self,index):
		"""
		Returns image name with extension.
		"""
		image_name = os.path.join(index + self._image_ext)
		return image_name
	def _get_widths(self):
		return [PIL.Image.open(self.image_path_at(i)).size[0]
				for i in range(self.num_images)]
	def _get_heights(self):
		return [PIL.Image.open(self.image_path_at(i)).size[1]
				for i in range(self.num_images)]
	def _get_dim(self, index):
		return PIL.Image.open(self.image_path_from_index(index)).size[0:2]
	def _get_dim_path(self, path):
		return PIL.Image.open(path).size[0:2]
	def _set_randome_seed(self,value):
		self._random_seed = value
#  Object detection helpers
	@property
	def roidb_handler(self):
		return self._roidb_handler
	@property
	def roidb(self):
		''' A roidb is a list of dictionaries, each with the following keys:
		boxes,   gt_overlaps, gt_classes, flipped'''
		if self._roidb is not None:
			return self._roidb
		self._roidb = self.roidb_handler()
		return self._roidb
	@property
	def get_list_of_images(self):
		return [self.image_path_from_index(idx) for idx in self._image_index]
	@roidb_handler.setter
	def roidb_handler(self, val):
		self._roidb_handler = val
	def gt_roidb(self):
		"""
		Return the database of ground-truth regions of interest.
		This function loads/saves from/to a cache file to speed up future calls.
		"""
		cache_file = join(self.cache_path, self._name + '_' + self._image_set+ '_gt_roi.pkl')
		if os.path.exists(cache_file) and not self._regen_pkl:
			with open(cache_file, 'rb') as fid:
				try:
					activities = pickle.load(fid)
				except:
					activities = pickle.load(fid, encoding='bytes')
			print('{} activities loaded from {}'.format(self._name, cache_file))
			return activities

		gt_roi = [self._load_annotation(index)
					for index in self._image_index]
		with open(cache_file, 'wb') as fid:
			pickle.dump(gt_roi, fid, pickle.HIGHEST_PROTOCOL)
		print('wrote gt roidb to {}'.format(cache_file))
		return gt_roi
	def gt_annotation_evaluation(self):
		"""
		return all images sorted according to image path.
		"""
		gt_annots = {self.image_path_from_index(index):
					 self._load_annotation_evaluation(index)
					for index in self._image_index}
		return gt_annots
	def _load_annotation(self, index):
		"""
		Load image and bounding boxes.
		"""
		annotation_file = os.path.join(self._jaad_path,self._image_set,
		'annotations_' + self._annotation_type, index + '.txt')
		assert os.path.exists(annotation_file), \
		'Path does not exist: {}'.format(annotation_set_folder)
		bbs = []
		with open(annotation_file) as f:
			annot_lines = [x.strip() for x in f.readlines()]
		for l in annot_lines:
			elements = l.split(' ')
		# check for pedestrian class tag, occlusion tag and height
			if elements[0].find('ped') != -1:
				if not self._use_occlusion and int(elements[5]):
					continue
				else:
					if int(elements[4]) > self._height_rng[0] and int(elements[4]) < self._height_rng[1]:
						bbs.append(elements)
		num_objs = len(bbs)
		boxes = np.zeros((num_objs, 4), dtype = np.uint16)
		gt_classes = np.zeros((num_objs), dtype = np.int32)
		overlaps = np.zeros((num_objs, self._num_classes), dtype = np.float32)
		tags = []
		occlusion = []
		# "Seg" area for pascal is just the box area
		seg_areas = np.zeros((num_objs), dtype = np.float32)
		# Load object bounding boxes into a data frameself.
		for ix, bbox in enumerate(bbs):
			if self._squarify_ratio > 0:
				bbox = self._squarify(bbox,index)
	 		# Make pixel indexes 0-based
			x1 = float(bbox[1]) - 1
			y1 = float(bbox[2]) - 1
			x2 = float(bbox[1]) + float(bbox[3]) - 1
			y2 = float(bbox[2]) + float(bbox[4]) - 1
			cls = 1 #self._class_to_ind[obj.find('name').text.lower().strip()]
			boxes[ix, :] = self._bbox_sanity_check([x1, y1, x2, y2],index) #[x1, y1, x2, y2]
			gt_classes[ix] = cls
			overlaps[ix, cls] = 1.0
			seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
			tags.append(bbox[0])
			occlusion.append(int(bbox[5]))
		overlaps = scipy.sparse.csr_matrix(overlaps)
		return {'tags': tags,
			'boxes': boxes,
			'gt_classes': gt_classes,
			'gt_overlaps': overlaps,
			'flipped': False,
			'seg_areas': seg_areas,
			'occlusion': occlusion}
	def _load_annotation_evaluation(self, index):
		"""
		Load image and bounding boxes.
		"""
		annotation_file = os.path.join(self._jaad_path,self._image_set,
		'annotations_' + self._annotation_type, index + '.txt')
		assert os.path.exists(annotation_file), \
		'Path does not exist: {}'.format(annotation_file)
		bbs = []
		with open(annotation_file) as f:
			annot_lines = [x.strip() for x in f.readlines()]
		for l in annot_lines:
			elements = l.split(' ')
		# check for pedestrian class tag, occlusion tag and height
			if elements[0].find('ped') != -1 or elements[0].find('people') != -1:
				bbs.append(elements)
		num_objs = len(bbs)
		boxes = np.zeros((num_objs, 4), dtype=np.uint16)
		gt_classes = np.zeros((num_objs), dtype=np.int32)
		# if the lables have occlusion or are below the allowable size
		difficult= np.zeros((num_objs), dtype=np.int32)
		overlaps = np.zeros((num_objs, self._num_classes), dtype=np.float32)
		tags = []
		# "Seg" area for pascal is just the box area
		seg_areas = np.zeros((num_objs), dtype=np.float32)
		# Load object bounding boxes into a data frameself.
		for ix, bbox in enumerate(bbs):
			# set a difficult as 1 if do not want to use occlusions
			if not self._use_occlusion and int(bbox[5]):
				difficult[ix] = 1
			if int(bbox[4]) < self._height_rng[0] or int(bbox[4]) > self._height_rng[1]:
				difficult[ix] = 1
			if bbox[0].find('people') != -1:
				difficult[ix] = 1

			# Squarify  bounding boxes to fix aspect retios, e.g. (w/h) = 0.41
			if self._squarify_ratio > 0:
				bbox = self._squarify(bbox,index)

			# Make pixel indexes 0-based
			x1 = float(bbox[1]) - 1
			y1 = float(bbox[2]) - 1
			x2 = float(bbox[1]) + float(bbox[3]) - 1
			y2 = float(bbox[2]) + float(bbox[4]) - 1
			cls = 1 #self._class_to_ind[obj.find('name').text.lower().strip()]
			boxes[ix, :] = self._bbox_sanity_check([x1, y1, x2, y2],index) #[x1, y1, x2, y2]
			gt_classes[ix] = cls
			overlaps[ix, cls] = 1.0
			seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
			tags.append(bbox[0])
		overlaps = scipy.sparse.csr_matrix(overlaps)
		return {'tags': tags,
			'boxes': boxes,
			'gt_classes': gt_classes,
			'gt_overlaps': overlaps,
			'flipped': False,
			'seg_areas': seg_areas,
			'difficult':difficult}
	def append_flipped_images(self):
		num_images = self.num_images
		widths = self._get_widths()
		for i in range(num_images):
			boxes = self.roidb[i]['boxes'].copy()
			oldx1 = boxes[:, 0].copy()
			oldx2 = boxes[:, 2].copy()
			boxes[:, 0] = widths[i] - oldx2 #- 1
			boxes[:, 2] = widths[i] - oldx1 #- 1
			assert (boxes[:, 2] >= boxes[:, 0]).all()
			entry = {'boxes': boxes,
					'gt_overlaps': self.roidb[i]['gt_overlaps'],
					'gt_classes': self.roidb[i]['gt_classes'],
					'flipped': True}
			self.roidb.append(entry)
		self._image_index = self._image_index * 2
	def _squarify(self,bbox, index):
		width_change = float(bbox[4])*self._squarify_ratio - float(bbox[3])
		bbox[1] = str(float(bbox[1]) - width_change/2)
		bbox[3] = str(float(bbox[3]) + width_change)
		# Squrify is applied to bounding boxes in Matlab coordinate starting from 1
		if float(bbox[1]) < 1:
			bbox[1] = '1'
		img_dimensions = self._get_dim(index)
		# check whether the new bounding box goes beyond image boarders
		# If this is the case, the bounding box is shifted back
		if float(bbox[1]) + float(bbox[3]) > img_dimensions[0]:
			bbox[1] = str(-float(bbox[3]) + img_dimensions[0])
		return bbox
	def _bbox_sanity_check(self,bbox,index):
		'''
		This is to confirm that the bounding boxes are within image boundaries.
		If this is not the case, modifications is applied.
		This is to deal with inconsistencies in the annotation tools
		'''
		img_dimensions = self._get_dim(index)
		if bbox[0] < 0:
			bbox[0] = 0.0
			print(bbox[0])
		if bbox[1] < 0:
			bbox[1] = 0.0
		if bbox[2] >= img_dimensions[0]:
			bbox[2] = float(img_dimensions[0]) - 1
		if bbox[3] >= img_dimensions[1]:
			bbox[3] = float(img_dimensions[1]) - 1
		return bbox
	def _get_jaad_results_file_template(self):
		filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
		path = os.path.join(
			self._jaad_path,
			'results',
			filename)
		return path
	def _write_jaad_results_file(self, all_boxes):
		for cls_ind, cls in enumerate(self.classes):
			if cls == '__background__':
				continue
			print('Writing {} jaad results file'.format(cls))
			filename = self._get_jaad_results_file_template().format(cls)
			with open(filename, 'wt') as f:
				for im_ind, index in enumerate(self.image_index):
					dets = all_boxes[cls_ind][im_ind]
					if dets == []:
						continue
          # the VOCdevkit expects 1-based indices
					for k in range(dets.shape[0]):
						f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
						format(index, dets[k, -1],
						dets[k, 0] + 1, dets[k, 1] + 1,
						dets[k, 2] + 1, dets[k, 3] + 1))

	# Methods to generate data for object detection
	# TODO: update the code to generate csv files for different classes of objects
	# TODO: Turn the list to np array?
	# TODO: add check path file exists
	def get_data_frcnn(self,image_set):

		self._image_set = image_set
		self._image_index = self._load_image_set_index(self._frame_skip)
		imgP = [self.image_path_from_index(img_p) for img_p in self._image_index]
		roidb = self.gt_roidb()
		classes_count = {}
		class_mapping = {}
		all_imgs = {}
		class_name = 'pedestrian'
		classes_count['bg'] = 0
		class_mapping['bg'] = 1
		classes_count[class_name] = 0
		class_mapping[class_name] = 0
		for imgPath, roi in zip(imgP, roidb):
			all_imgs[imgPath] = {}
			# img = cv2.imread(imgPath)
			# (rows,cols) = img.shape[:2]
			(cols,rows) = self._get_dim_path(imgPath)
			all_imgs[imgPath]['filepath'] = imgPath
			all_imgs[imgPath]['width'] = cols
			all_imgs[imgPath]['height'] = rows
			all_imgs[imgPath]['bboxes'] = []
			for box in roi['boxes']:
				all_imgs[imgPath]['bboxes'].append({'class': class_name, 'x1': box[0],
				'x2' :box[2], 'y1': box[1], 'y2':box[3]})
			classes_count['pedestrian'] += len(roi['boxes'])
		all_data = []
		for key in all_imgs:
			all_data.append(all_imgs[key])
		return all_data, classes_count, class_mapping
	def generate_csv_data_retinanet(self,image_set,filePath, mapping = False):
		self._image_set = image_set
		self._image_index = self._load_image_set_index(self._frame_skip)
		imgP = [self.image_path_from_index(img_p) for img_p in self._image_index]
		roidb = self.gt_roidb()
		class_name = 'pedestrian'
		with open(filePath + '.csv', "w") as f:
			for imgPath, roi in zip(imgP, roidb):
				if not roi['boxes'].any():
					f.write('%s,,,,,\n'% (imgPath))
				else:
					for box in roi['boxes']:
						f.write('%s,%.0f,%.0f,%.0f,%.0f,%s\n'% (imgPath,box[0],box[1],box[2],box[3], class_name))
		if mapping:
			map_path = filePath + '_mapping.csv'
			with open(map_path, "w") as f:
				f.write('%s,0\n'% (class_name))
	def generate_csv_data_yolo3(self,image_set,filePath, mapping = False):
		self._image_set = image_set
		self._image_index = self._load_image_set_index(self._frame_skip)
		imgP = [self.image_path_from_index(img_p) for img_p in self._image_index]
		roidb = self.gt_roidb()
		class_name = 'pedestrian'
		with open(filePath, "w") as f:
			for imgPath, roi in zip(imgP, roidb):
				if roi['boxes'].any():
					f.write('%s '% (imgPath))
					for box in roi['boxes']:
						f.write('%.0f,%.0f,%.0f,%.0f,%.0f '% (box[0],box[1],box[2],box[3], 0))
					f.write('\n')

		if mapping:
			with open('mapping_'+ filePath, "w") as f:
				f.write('%s,0\n'% (class_name))
	def generate_csv_data_ssd(self,image_set,filePath):
		self._image_set = image_set
		self._image_index = self._load_image_set_index(self._frame_skip)
		imgP = [self.image_path_from_index(img_p) for img_p in self._image_index]
		roidb = self.gt_roidb()
		class_name = 'pedestrian'
		with open(filePath, "w") as f:
			for imgPath, roi in zip(imgP, roidb):
				for box in roi['boxes']:
					f.write('%s,%.0f,%.0f,%.0f,%.0f,%.0f\n'% (imgPath,box[0],box[1],box[2],box[3], 1))

	# Helper functions for behavior data
	def _get_video_id_from_image_index(self,path):
		'''extracts video id from the image name'''
		vidKey = path.split('/')
		return vidKey[0]
	def _get_frame_number(self,path):
		'''extracts frame number from the image name'''
		fnum = path.split('/I')
		return int(fnum[1][0:]) + 1 # frame numbers in extracted jaad folder start from 0
	def _get_video_number(self,vid_id):
		'''extracts frame number from the image name'''
		fnum = vid_id.split('_')
		return int(fnum[1][0:]) # frame numbers in extracted jaad folder start from 0
	def _load_video_ids(self):
		"""
		Load the indexes listed in this dataset's image set file.
		"""
		video_folders = os.path.join(self._jaad_path, self._image_set, 'images')
		assert os.path.exists(video_folders), \
		'Path does not exist: {}'.format(video_folders)
		vidoe_ids = [f for f in sorted(listdir(video_folders))
					if isdir(os.path.join(video_folders, f))]
		return vidoe_ids
	def _get_pedestrian_activities_image(self,image_name):
		# Extract the video name from the image name
		vidKey = self._get_video_id_from_image_index(image_name)
		framenum = self._get_frame_number(image_name)
		# generate a xml tree from the behavioral tag
		tree = ET.parse(self._behaviral_path_from_index_xml(vidKey))
		subjs = [t.tag for t in tree.find('actions') if t.tag not in 'Driver']
		# activities = []
		activities = {}
		for s in subjs:
			t = {s:[act.get('id')
					for act in tree.find('actions').find(s)
					if framenum >= int(act.get('start_frame')) and
					framenum <= int(act.get('end_frame'))]}

			if  t[s]:
				activities.update(t)
		return(activities)
	def _get_driver_activities_image(self,image_name):
		# Extract the video name from the image name
		vidKey = self._get_video_id_from_image_index(image_name)
		framenum = self._get_frame_number(image_name)
		# generate a xml tree from the behavioral tag
		tree = ET.parse(self._behaviral_path_from_index_xml(vidKey))
		activities = [act.get('id')
				for act in tree.find('actions').find('Driver')
				if framenum >= int(act.get('start_frame')) and
				framenum <= int(act.get('end_frame'))]
		return activities
	def _generate_labels_for_pedestrian_activities(self,ped_actions):
		'''
		labels: motion(0/1), looking(0/1), handwave(0/1),nod(0/1), reaction(0-2),
		 crossing(0/1)
		'''
		# motion_labels = ['standing','walking']
		# gesture_labels = ['looking','handwave','nod']
		# crossing_labels = ['crossing']
		# reaction_labels = ['clear path','speed up','slow down']
		# all_ped_labels = motion_labels + gesture_labels + crossing_labels + reaction_labels
		label_map = self._get_activity_map_lable_to_index()
		ped_labels = {}
		for ped in sorted(ped_actions.keys()):
			ped_labels[ped] = [0]*6
			ped_labels[ped][0] = 1 # sets the walking tag to 1 unless standing tag is observed
			for act in ped_actions[ped]:
				idx = label_map[act.lower()][0]
				val = label_map[act.lower()][1]
				ped_labels[ped][idx] = val
		return ped_labels
	def _get_activity_map_lable_to_index(self):
		'''
		maps the activity labels to indecies and values in gt vector
		'''
		map_labels = {
		'standing': [0,0],
		'walking': [0,1],
		'looking':[1,1],
		'handwave':[2,1],
		'nod':[3,1],
		'crossing':[4,1],
		'clear path':[5,1],
		'speed up':[5,2],
		'slow down':[5,3]
		}
		return map_labels
	def _generate_labels_for_driver_activities(self,dr_actions):
		'''
		labels: driver(0-4)
		'''
		driver_labels = ['moving slow', 'moving fast', 'decelerating', 'accelerating','stopped']

		return [driver_labels.index(dr_actions[0])]
	def _behaviral_path_from_index_xml(self, index):
		"""
		Construct an image path from the image's "index" identifier.
		"""
		beh_path = os.path.join(self._jaad_path, 'behavioral_data_xml', index + '.xml')
		assert os.path.exists(beh_path), \
		'Path does not exist: {}'.format(beh_path)
		return beh_path
	def _get_pedestrian_activities_video(self,video_id):
		# Extract the video name from the image name
		vidKey = self._get_video_id_from_image_index(video_id)
		# make a list of behavioral tags for each video
		self._video_ids = self._load_video_ids()
			# create a full path to the file for each video id
		beh_full_path = {beh_p : self._behaviral_path_from_index_xml(beh_p)
						for beh_p in self._video_ids}
		# generate a xml tree from the behavioral tag
		tree = ET.parse(beh_full_path[vidKey])
		subjs = [t.tag for t in tree.find('actions') if t.tag not in 'Driver']
		activities = []
		for s in subjs:
			t = {'subject':s,
				'video_id':vidKey,
				'actions':[{'start':int(act.get('start_frame')),
							'end':int(act.get('end_frame')),
							'action':act.get('id')}
							for act in tree.find('actions').find(s)]}
			activities.append(t)
		return(activities)
	def _get_pedestrian_bbox(self,box):
		'''Returns the bounding box of pedestrians'''
		bbox = {}
		for b,t,o in zip(box['boxes'],box['tags'], box['occlusion']):
			# Check whether there is a pedestrian tags on bounding boxes
			if t.find('pedestrian') != -1:
				# Split is for the ones that have
				# separated ids, e.g. pedestrians_p1 and pedestrian_p2
				ped_id = t.split("_")[0]
				bbox[ped_id] = np.append(b,o)
		return bbox

	def _crop_samples(self,box,data_size,imPath):
		'''Crops out all bounding boxes from a given image '''
		array_samples = {}
		img_data = image_keras.load_img(imPath)
		for ped in sorted(box.keys()):
			# crop the pedestrian sample from the image
			cropped_sample = img_data.crop(box[ped][0:4])
			# resize the image to the size of the network input
			resied_sample = cropped_sample.resize(data_size,PIL.Image.NEAREST)
			# convert the PIL image to np array
			array_samples[ped] =  image_keras.img_to_array(resied_sample)
		return array_samples
	def _crop_sample(self,box,data_size,imPath):
		'''Crops out a single bounding box from the given image'''
		array_samples = {}
		img_data = image_keras.load_img(imPath)
		# crop the pedestrian sample from the image
		cropped_sample = img_data.crop(box[0:4])
		# resize the image to the size of the network input
		resied_sample = cropped_sample.resize(data_size,PIL.Image.NEAREST)
		# convert the PIL image to np array
		array_sample[ped] =  image_keras.img_to_array(resied_sample)
		return array_sample
	def _initialize_dict_behavior(self,video_id):
		'''Generates an empty dictionary for behavior database'''
		peds = {}
		# Extract the video name from the image name
		beh_file_path =  self._behaviral_path_from_index_xml(video_id)
		# generate a xml tree from the behavioral tag
		tree = ET.parse(beh_file_path)
		subjs = [t.tag for t in tree.find('actions') if t.tag not in 'Driver']
		for s in subjs:
			id =  s.split("_")
			ped_id = id[0] # for the ones with two parts
			is_two_parts = int(len(id) > 1)
			peds[ped_id] = {
			'images':[],
			'bboxes':[],
			'actions':[],
			'actions_gts': [],
			'driver_actions':[],
			'driver_gts':[],
			'samples':[],
			'frames':[],
			'is_crossing': -1,
			'decision_frame':-1,
			'two_parts': is_two_parts
			}
		return peds
	def _get_ped_decision_point(self):
		"""
		Generates data for pedestrian.
		output: 'is_crossing': whether pedestrians will cross (1), not cross (0) or irrelevant
		(already crossing or no decision)(-1)
			    'decision_frame': at what point (frame number) pedestrian will decide whether to cross or not.
				if set to -1, the pedestrian is either irrelevant or alraedy crossing (from the moment it appears)
		"""
		decision_file = os.path.join(self._jaad_path,'jaad_decision_points.csv')
		assert os.path.exists(decision_file), \
		'Path does not exist: {}'.format(decision)
		dec_points = {}
		with open(decision_file) as f:
			decision_lines = [x.strip() for x in f.readlines()]
		for l in decision_lines:
			elements = l.split(',')
			vid = "video_%04.0f" % int(elements[0])
			if vid in dec_points.keys():
				dec_points[vid][elements[1]] = {'is_crossing':int(elements[2]),'decision_frame':int(elements[3])}
			else:
				dec_points[vid] ={elements[1]:{'is_crossing':int(elements[2]),'decision_frame':int(elements[3])}}

		return dec_points

	# Generate data for images
	def generate_data_acitivity_stats(self,	fskip = 30, labels = ["walking"]):
		'''Generates statistics of the behvaior dataset'''
		activities = self.generate_database_activity(frame_skip = fskip)
		label_maps = self._get_activity_map_lable_to_index()
		get_labels = set([label_maps[t][0] for t in labels])
		num_pedestrians = 0
		total_number_of_samples = 0
		num_samples = [0]*len(labels)
		gt_labels = []
		for vid in sorted(activities.keys()):
			for pid in sorted(activities[vid].keys()):
				num_pedestrians += 1
				for idx in range(len(activities[vid][pid]['images'])):
					total_number_of_samples += 1
					# print(activities[vid][pid]['images'][idx])
					# print(activities[vid][pid]['bboxes'][idx])
					gt_labels.append([activities[vid][pid]['actions_gts'][idx][t]
					for t in get_labels])
					# print(gt_labels)
					count_labels = [1 if (p in activities[vid][pid]['actions'][idx]) or
					(p == "walking" and "standing" not in activities[vid][pid]['actions'][idx])
					else 0 for p in labels]
					num_samples = list(map(add, num_samples, count_labels))
		print('---------------------------------------------------------')
		print("Number of videos: %d" % len(activities.keys()))
		print("Number of unique pedestrians: %d" % num_pedestrians)
		print("Labels selected and count: ")
		print('\n'.join('{} = {}'.format(l,s) for l,s in zip(labels,num_samples)))
		print('Total Number of samples = %d'%total_number_of_samples)
	def generate_data_environment(self, tags = ['weather','location'],data_size = (224,224)):
		''' Generates examples for environment images.
		possible tags: ['time_of_day', 'weather','location']'''
		# A helper function to convert ground truth tags to one-hot vectors
		mlb = MultiLabelBinarizer()
		# make a list of behavioral tags for each video
		self._video_ids = self._load_video_ids()
		# which tags will be included in the data
		env_tags = tags
		# create a full path to the file for each video id
		beh_full_path = {beh_p : self._behaviral_path_from_index_xml(beh_p)
						for beh_p in self._video_ids}
		# create a full path to the file for each image
		img_full_path = [[img_p ,self.image_path_from_index(img_p)]
		 				for img_p in self._image_index]
		gts = []
		img_data = np.zeros((len(img_full_path), data_size[0],data_size[1],3), dtype=np.float32)
		for i,imPath in enumerate(img_full_path):
			# Extract the video name from the image name
			vidKey = self._get_video_id_from_image_index(imPath[0])
			# generate a xml tree from the behavioral tag
			tree = ET.parse(beh_full_path[vidKey])
			tags = list(tree.find('tags'))
			cond_gt = [t.attrib["val"]  for t in tags if t.tag in env_tags]
			gts.append(cond_gt)
			img_data[i] = image_keras.img_to_array(image_keras.load_img(imPath[1],
			target_size= data_size))
		gts_one_hot = mlb.fit_transform(gts)
		return {'images': img_data,
				'ground_truth': gts_one_hot,
				'classes': mlb.classes_}
		# TODO: Test it with a network training
	def generate_data_activity_static(self, image_set,data_size = (224,224),
	 label_transform_params = {}, just_ground_truth = False):
		'''Generates activity data per image. Suitable for basic acitivties such as
		looking or walking actions.'''
		# A helper function to convert ground truth tags to one-hot vectors
		ped_mlb= MultiLabelBinarizer()
		dr_mlb = MultiLabelBinarizer()
		# create a full path to the file for each image
		ped_gts = []
		ped_bbox = []
		driver_gts = []
		data_all = []
		image_paths = []
		# make a list of behavioral tags for each video
		self._video_ids = self._load_video_ids()
		self._image_set = image_get
		image_index = self._load_image_set_index(self._frame_skip)
		for index in image_index:
			#read bounding box annotations
			box = self._load_annotation(index)
			if not box['boxes'].any():
				continue
			# read images full path from indecies
			imPath = self.image_path_from_index(index)
			if not just_ground_truth:
				# load image in PIL format
				img_data = image_keras.load_img(imPath)
			# load activities of pedestrians for the given image
			ped_acts = self._get_pedestrian_activities_image(index)
			dr_acts = self._get_driver_activities_image(index)
			for b,t in zip(box['boxes'],box['tags']):
				# Check whether there is a pedestrian tags on bounding boxes
				if t.find('pedestrian') != -1:
					# Load the activities for a given pedestrian. Split is for the ones that have
					# separated ids, e.g. pedestrians_p1 and pedestrian_p2
					ped_actions = ped_acts[t.split('_')[0]]
					ped_actions  = self._transform_pedestrian_activity_labels_for_static(ped_actions,**label_transform_params)
					if not just_ground_truth:
						# crop the pedestrian sample from the image
						imData = img_data.crop(b)
						# resize the image to the size of the network input
						imData = imData.resize(data_size,PIL.Image.NEAREST)
						# convert the PIL image to np array
						imData = image_keras.img_to_array(imData)
						data_all.append(imData)
					ped_gts.append(ped_actions)
					driver_gts.append(dr_acts)
					ped_bbox.append(b)
					image_paths.append(imPath)

		# Convert text ground truths to one-hot vectors
		ped_gts =  ped_mlb.fit_transform(ped_gts)
		driver_gts = dr_mlb.fit_transform(driver_gts)
		if not just_ground_truth:
			assert(len(data_all) == len(ped_gts)),"number of pedestrian ground truth\
			 %.0f does not match data %.0f: \n" % (len(ped_gts),len(data_all))
			assert(len(data_all) == len(driver_gts)),"number of driver ground truth\
			 %.0f does not match data %.0f: \n" % (len(driver_gts),len(data_all))
		print('---------------------------------------------------------')
		print('Number of data samples: %.i' % (len(data_all)))
		print('Pedestrian activities: {}'.format(ped_mlb.classes_))
		print('Pedestrian activity classes Count: {}'.format(np.sum(ped_gts,axis=0)))
		print('Driver activities: {}'.format(dr_mlb.classes_))
		print('Driver activity classes Count: {}'.format(np.sum(driver_gts,axis=0)))

		return {'images': np.array(data_all),
				'images_path' : image_paths,
				'ped_bbox': ped_bbox,
				'ped_actions': ped_gts,
				'ped_act_classes': ped_mlb.classes_,
				'driver_actions': driver_gts,
				'driver_act_classes': dr_mlb.classes_}

	# @profile

	def _get_action_sequence_with_optical_flow(self,params, activities):
		'''
		Generates sequence data for a given action. Each data point (frame) has
		an associated optical flow element which contains a seuqence of
		images (and bounding boxes) that include frames before and after
		(specified by optical flow window)	the current frame.
		'''
		print('---------------------------------------------------------')
		print("Generating data sequences for %s" % params['label'])

		label_maps = self._get_activity_map_lable_to_index()
		# label map is a dic: key = activity, val[0]= index of the gt in the gt generated by database.
		# val[1] = value, e.g. walking/standing is at index 0, and value of standing is 0 and walking 1
		label_idx = label_maps[params['label']][0]
		num_pedestrians = 0
		frame_stride = params['fstride']
		optical_window = 2 if params['optical_flow_window'] < 2 else params['optical_flow_window']
		if optical_window/2 > frame_stride:
			print('optical flow window size %d is set bigger than frame stride %d * 2.\n \
			The window is set to %d' %(optical_window ,frame_stride,frame_stride * 2))
			optical_window = frame_stride * 2

		# select the number of frames before and after a sequence. e.g. consider the current fram T.
		#if optical flow is an odd number such as 3 the cut out would be [T-1,T,T+1] and
		# if the optical window is an even number, e.g. 4, the cut out would be [T-2,T-1, T, T + 1]
		num_frames_before = optical_window//2
		num_frames_after = num_frames_before - 1 + optical_window % 2

		gt_data = []
		bbox = []
		gt_labels = []
		optical_flow = []
		optical_flow_bbox = []
		frame_count = []
		# Generates image sequences and corresponding optical flow windows.
		#  It adds an image to the sequence until the tag for that
		# label changes, e.g. from walking to standing. At this point the sequence is added to the
		#vecotr and a new sequence is formed.
		for vid in sorted(activities.keys()):
			for pid in sorted(activities[vid].keys()):
				if not activities[vid][pid]['images']:
					continue
				num_pedestrians += 1
				last_label = activities[vid][pid]['actions_gts'][0][label_idx];
				x = []
				b = []
				opt_flow = []
				opt_flow_b = []
				for idx in range(len(activities[vid][pid]['images'])):
					img_path = activities[vid][pid]['images'][idx]
					bounding_box = activities[vid][pid]['bboxes'][idx]
					next_label = activities[vid][pid]['actions_gts'][idx][label_idx]
					if last_label != next_label:
						if len(x) > optical_window:
							trunc_x = x[num_frames_before:len(x)-num_frames_after]
							if len(trunc_x)/frame_stride > params['min_cutoff']:
								trunc_b = b[num_frames_before:len(b)-num_frames_after]
								gt_data.append(trunc_x[::frame_stride])
								bbox.append(trunc_b[::frame_stride])
								frame_count.append(len(trunc_x[::frame_stride]))
								opt_flow = [x[i:i + optical_window] for i in range(len(x) - optical_window + 1 )]
								opt_flow_b = [b[i:i + optical_window] for i in range(len(b) - optical_window + 1)]
								optical_flow.append(opt_flow[::frame_stride])
								optical_flow_bbox.append(opt_flow_b[::frame_stride])
								gt_labels.append(last_label)
						last_label = next_label
						x = []
						b = []
						opt_flow = []
						opt_flob_b = []
					x.append(img_path)
					b.append(bounding_box)
				if len(x) > optical_window:
					trunc_x = x[num_frames_before:len(x)-num_frames_after]
					if len(trunc_x)/frame_stride > params['min_cutoff']:
						trunc_b = b[num_frames_before:len(b)-num_frames_after]
						gt_data.append(trunc_x[::frame_stride])
						bbox.append(trunc_b[::frame_stride])
						frame_count.append(len(trunc_x[::frame_stride]))
						opt_flow = [x[i:i + optical_window] for i in range(len(x)-optical_window + 1)]
						opt_flow_b = [b[i:i + optical_window] for i in range(len(b)- optical_window + 1)]
						optical_flow.append(opt_flow[::frame_stride])
						optical_flow_bbox.append(opt_flow_b[::frame_stride])
						gt_labels.append(last_label)

		# TODO: only works for binary at the moment. e.g. reactions should be dealt differently
		num_pos_samples = np.count_nonzero(np.array(gt_labels))
		print('Number of pedestrians: %d ' % num_pedestrians )
		print('Subset: %s'% self._image_set)
		print('Sample Count: Positive: %d  Negative: %d '
		% (num_pos_samples, len(gt_labels)-num_pos_samples))
		print('Sequence length: Min %d, Max %d, Avg. %0.2f\n' % (min(frame_count),
		 max(frame_count), sum(frame_count)/len(frame_count)))
		return {'gt_data': gt_data,
				'bbox': bbox,
				'optical_flow':optical_flow,
				'optical_flow_bbox':optical_flow_bbox,
				'gt_labels': gt_labels},frame_count
	def _adjust_seq_length(self,params, seq_data,frame_count):
		'''
		This method adjusts the length of sequences.
		Options are, 'trunc' truncates long sequences to the maximum size, 'divide' breaks the
		long sequences to smaller sizes no longer than maximum size, 'none' returns the original sequences
		'seq_overlap_rate' if 'divide' is selected, this defines how much divided sequences can overlap
		'''
		print('---------------------------------------------------------')
		# Generate seuqences with given stride and min cut off
		gt_data = seq_data['gt_data']
		bbox = seq_data['bbox']
		gt_labels = seq_data['gt_labels']
		optical_flow = seq_data['optical_flow']
		optical_flow_bbox = seq_data['optical_flow_bbox']
		max_size = params['max_size']
		new_gt_data = []
		new_bbox = []
		new_gt_labels = []
		new_optical_flow = []
		new_optical_flow_bbox = []
		overlap_stride = max_size if params['seq_overlap_rate'] == 0 else int((1 - params['seq_overlap_rate'])*max_size)
		if params['seq_gen_option'] != 'none':
			print("Adjusting the length of sequences using %s method"% params['seq_gen_option'])
			if max_size == 0:
				max_size = sum(frame_count)/len(frame_count)
			for i in range(0, len(gt_data)):
				if params['seq_gen_option'] == 'trunc':
					if len(gt_data[i]) > max_size:
						gt_data[i] = gt_data[i][0:max_size]
						bbox[i] = bbox[i][0:max_size]
						optical_flow[i] = optical_flow[i][0:max_size]
						optical_flow_bbox[i] = optical_flow_bbox[i][0:max_size]
					new_gt_data.append(gt_data[i])
					new_bbox.append(bbox[i])
					new_optical_flow.append(optical_flow[i])
					new_optical_flow_bbox.append(optical_flow_bbox[i])
					new_gt_labels.append(gt_labels[i])
				elif params['seq_gen_option'] == 'random':
					if len(gt_data[i]) > max_size:
						np.random.seed(self._random_seed)
						keep_indecies = np.random.np.random.choice(len(gt_data[i]),max_size,replace=False)
						gt_data[i] = [gt_data[i][j] for j in  keep_indecies]
						bbox[i] = [bbox[i][j] for j in keep_indecies]
						optical_flow[i] = [optical_flow[i][j] for j in keep_indecies]
						optical_flow_bbox[i] = [optical_flow_bbox[i][j] for j in keep_indecies]
					new_gt_data.append(gt_data[i])
					new_bbox.append(bbox[i])
					new_optical_flow.append(optical_flow[i])
					new_optical_flow_bbox.append(optical_flow_bbox[i])
					new_gt_labels.append(gt_labels[i])
				elif params['seq_gen_option'] == 'divide':
					if len(gt_data[i])> max_size:
						split_gt = [gt_data[i][x : x + max_size] for x in range(0,len(gt_data[i]),overlap_stride)]
						split_bbox = [bbox[i][x : x + max_size] for x in range(0,len(bbox[i]),overlap_stride)]
						split_optical_flow = [optical_flow[i][x : x + max_size] for x in range(0,len(optical_flow[i]),overlap_stride)]
						split_optical_flow_bbox = [optical_flow_bbox[i][x: x + max_size] for x in range(0,len(optical_flow_bbox[i]),overlap_stride)]
						if len(split_gt[-1]) < params['min_cutoff'] :
							split_gt = split_gt[:-1]
							split_bbox = split_bbox[:-1]
							split_optical_flow = split_optical_flow[:-1]
							split_optical_flow_bbox = split_optical_flow_bbox[:-1]
						new_gt_data.extend(split_gt)
						new_bbox.extend(split_bbox)
						new_optical_flow.extend(split_optical_flow)
						new_optical_flow_bbox.extend(split_optical_flow_bbox)
						new_gt_labels.extend([gt_labels[i]]*len(split_gt))
					else:
						new_gt_data.append(gt_data[i])
						new_bbox.append(bbox[i])
						new_optical_flow.append(optical_flow[i])
						new_optical_flow_bbox.append(optical_flow_bbox[i])
						new_gt_labels.append(gt_labels[i])
			num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
			print('New sample count after adjustment:\t Positive: %d \t Negative: %d\n'
			% (num_pos_samples, len(new_gt_labels)-num_pos_samples))
		else:
			max_size = max(frame_count)
			new_gt_data = gt_data
			new_bbox = bbox
			new_optical_flow = optical_flow
			new_optical_flow_bbox = optical_flow_bbox
			new_gt_labels = gt_labels


		return {'gt_data': new_gt_data,
				'bbox': new_bbox,
				'optical_flow':new_optical_flow,
				'optical_flow_bbox':new_optical_flow_bbox,
				'gt_labels': new_gt_labels}
	def _remove_occluded_samples(self,params,seq_data):
		'''
		Removes the sequence samples that have more occluded frames more than a specified threshold
		(occlusion_removal_percent).
		Options (occlusion_removal_type) are: 'absolute', percentage calculated with respect to the max size,
		'relative' percentage calculated based on the average length of the sequences.
		'''
		gt_data = seq_data['gt_data']
		bbox = seq_data['bbox']
		gt_labels = seq_data['gt_labels']
		optical_flow = seq_data['optical_flow']
		optical_flow_bbox = seq_data['optical_flow_bbox']

		new_gt_data = []
		new_bbox = []
		new_gt_labels = []
		new_optical_flow = []
		new_optical_flow_bbox = []

		# removes seuqnces with occlusion
		if params['occlusion_removal_percent'] !=0:
			print('---------------------------------------------------------')
			print("Removing sequences with more than %.2f %s occlusion" \
			%(params['occlusion_removal_percent'], params['occlusion_removal_type']))
			if params['occlusion_removal_type'] not in ['absolute','relative']:
				raise Exception('Wrong occlusion removal type is selected.\n Options: absolute or relative')
			del_indecies = []
			for i,b in enumerate(bbox):
				count = 0
				for img_box in b:
					if img_box[4] == 1: # reads the occ tag from bounding box
						count += 1
				if params['occlusion_removal_type'] == 'absolute':
					if count/params['max_size'] >= params['occlusion_removal_percent']:
						del_indecies.append(i)
				elif params['occlusion_removal_type'] == 'relative':
					if count/len(b) >= params['occlusion_removal_percent']:
						del_indecies.append(i)
			new_gt_data = [gt_data[i] for i in range(0,len(gt_data)) if i not in del_indecies]
			new_bbox = [bbox[i] for i in range(0,len(bbox)) if i not in del_indecies]
			new_gt_labels = [gt_labels[i] for i in range(0,len(gt_labels)) if i not in del_indecies]
			new_optical_flow = [optical_flow[i] for i in range(0,len(optical_flow)) if i not in del_indecies]
			new_optical_flow_bbox = [optical_flow_bbox[i] for i in range(0,len(optical_flow_bbox)) if i not in del_indecies]

			num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
			print('New sample count after occlusion removal:\t Positive: %d  \t Negative: %d\n'
			% (num_pos_samples, len(new_gt_labels)-num_pos_samples))
			return {'gt_data': new_gt_data,
					'bbox': new_bbox,
					'optical_flow':new_optical_flow,
					'optical_flow_bbox':new_optical_flow_bbox,
					'gt_labels': new_gt_labels}
		else:
			print("No occlusion removal has been performed.")
			return seq_data
	def _balance_samples_count(self,params,seq_data):
		'''
		Balances the number of positive and negative samples.
		It reduces the size of the larger sequence to the smaller one
		balance: if true, it balances the number of positive and negative samples
		'''
		# balances the number of positive and negative samples
		print('---------------------------------------------------------')
		if params['balance']:
			print("Balancing the number of positive and negative samples")
			gt_data = seq_data['gt_data']
			bbox = seq_data['bbox']
			gt_labels = seq_data['gt_labels']
			optical_flow = seq_data['optical_flow']
			optical_flow_bbox = seq_data['optical_flow_bbox']
			num_pos_samples = np.count_nonzero(np.array(gt_labels))
			num_neg_samples = len(gt_labels)-num_pos_samples
			# finds the indecies of the samples with larger quantity
			if num_neg_samples == num_pos_samples:
				print('Positive and negative samples are already balanced')
				return seq_data
			else:
				if num_neg_samples > num_pos_samples:
					rm_index = np.where(np.array(gt_labels) == 0)[0]
				else:
					rm_index = np.where(np.array(gt_labels) == 1)[0]
				# Calculate the difference of sample counts
				dif_samples = abs(num_neg_samples - num_pos_samples)
				# shuffle the indecies
				np.random.seed(self._random_seed)
				np.random.shuffle(rm_index)
				# reduce the number of indecies to the difference
				rm_index = rm_index[0:dif_samples]
				# update the data
				new_gt_data = [gt_data[i] for i in range(0,len(gt_data)) if i not in rm_index]
				new_bbox = [bbox[i] for i in range(0,len(bbox)) if i not in rm_index]
				new_optical_flow = [optical_flow[i] for i in range(0,len(optical_flow)) if i not in rm_index]
				new_optical_flow_bbox = [optical_flow_bbox[i] for i in range(0,len(optical_flow_bbox)) if i not in rm_index]
				new_gt_labels = [gt_labels[i] for i in range(0,len(gt_labels)) if i not in rm_index]

				num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
				print('balanced:\t Positive: %d  \t Negative: %d\n'
				% (num_pos_samples, len(new_gt_labels)-num_pos_samples))
				return {'gt_data': new_gt_data,
						'bbox': new_bbox,
						'optical_flow':new_optical_flow,
						'optical_flow_bbox':new_optical_flow_bbox,
						'gt_labels': new_gt_labels}
		else:
			print("No sample balancing has been performed.")
			return seq_data
	def _get_crossing_sequence_with_optical_flow(self,params, activities):
		'''
		Generates sequence data for a given action. Each data point (frame) has an associated optical flow element
		which contains a seuqence of images (and bounding boxes) that include frames before and after (specified by optical flow window)
		the current frame.
		'''
		print('---------------------------------------------------------')
		print("Generating data sequences for crossing")

		# TODO: check the optical flow samples. It goes over entire samples even after decision frame
		num_pedestrians = 0
		optical_window = 2 if params['optical_flow_window'] < 2 else params['optical_flow_window']
		stride_mode = params['use_fixed_stride']
		seq_stride = params['fstride']
		# select the number of frames before and after a sequence. e.g. consider the current fram T.
		#if optical flow is an odd number such as 3 the cut out would be [T-1,T,T+1] and
		# if the optical window is an even number, e.g. 4, the cut out would be [T-2,T-1, T, T + 1]
		num_frames_before = optical_window//2
		num_frames_after = num_frames_before - 1 + optical_window % 2
		gt_data = []
		bbox = []
		gt_labels = []
		optical_flow = []
		optical_flow_bbox = []
		frame_count = []
		# Generates image sequences and corresponding optical flow windows.
		#  It adds an image to the sequence until the tag for that
		# label changes, e.g. from walking to standing. At this point the sequence is added to the
		#vecotr and a new sequence is formed.
		for vid in sorted(activities.keys()):
			for pid in sorted(activities[vid].keys()):
				if not activities[vid][pid]['images'] or activities[vid][pid]['decision_frame'] == -1 \
				or activities[vid][pid]['is_crossing'] == -1:
					continue
				# Check if the pedestrian is relevant
				dec_frame = activities[vid][pid]['decision_frame']
				try:
					frame_index = activities[vid][pid]['frames'].index(dec_frame)
				except:
					print("Decision frame for %s in %s is out of bound due to scale limitation." %(pid,vid))
					continue

				seq = activities[vid][pid]['images']
				box = activities[vid][pid]['bboxes']
				trunc_seq = seq[num_frames_before:frame_index-num_frames_after]

				if len(trunc_seq) <= params['min_cutoff']:
					continue
				trunc_bbox = box[num_frames_before:frame_index-num_frames_after]
				opt_flow = [seq[i:i + optical_window] for i in range(len(seq)- optical_window)]
				opt_flow_b = [box[i:i + optical_window] for i in range(len(box)- optical_window)]
				if not stride_mode:
					seq_stride = 1 if len(trunc_seq) < params['max_size'] else len(trunc_seq)//params['max_size']
				num_pedestrians += 1
				gt_data.append(trunc_seq[::seq_stride])
				bbox.append(trunc_bbox[::seq_stride])
				frame_count.append(len(trunc_seq[::seq_stride]))
				optical_flow.append(opt_flow[::seq_stride])
				optical_flow_bbox.append(opt_flow_b[::seq_stride])
				gt_labels.append(activities[vid][pid]['is_crossing'])


		num_pos_samples = np.count_nonzero(np.array(gt_labels))
		print('Number of pedestrians: %d ' % num_pedestrians )
		print('Subset: %s'% self._image_set)
		print('Sample Count: Positive: %d  Negative: %d '
		% (num_pos_samples, len(gt_labels)-num_pos_samples))
		print('Sequence length: Min %d, Max %d, Avg. %0.2f\n' % (min(frame_count),
		 max(frame_count), sum(frame_count)/len(frame_count)))
		return {'gt_data': gt_data,
				'bbox': bbox,
				'optical_flow':optical_flow,
				'optical_flow_bbox':optical_flow_bbox,
				'gt_labels': gt_labels},frame_count
	def get_opt_flow_image_list(self, image_set, opt_flow_algorithm, save_path, **opts):
		#generate text file with image pairs for offline optical flow computation
		#save it to save_path
		self._image_set = image_set

		params = {'fstride': 30, 'label': "walking",
		'min_cutoff': 3, 'max_size': 10, 'seq_gen_option':'trunc',
		'occlusion_removal_percent':0.2,'seq_overlap_rate': 0.50,
		'occlusion_removal_type':'absolute','balance': True,
		'optical_flow_window': 10,'use_fixed_stride': True}
		for i in opts.keys():
			params[i] = opts[i]


		fid = open(save_path, 'w')

		activities = self.generate_database_activity(self._image_set,frame_skip = 1)
		if params['label'] == 'crossing':
			sequence_data,frame_count = self._get_crossing_sequence_with_optical_flow(params, activities)
		else:
			sequence_data,frame_count = self._get_action_sequence_with_optical_flow(params, activities)

		print(sequence_data['optical_flow'])

		for seq in sequence_data['optical_flow']:
			for img_pair in seq:
				opt_flow_name = os.path.splitext(img_pair[1])[0].replace('images', 'opt_flow/' + opt_flow_algorithm) + '.flo'
				fid.write(img_pair[0] + ' ' + img_pair[1] + ' ' + opt_flow_name + '\n')
		print(save_path)
		fid.close()
		sys.exit(1)

	def generate_data_action_sequence(self,image_set,**opts):
		''' Generates activity seuqences:
		fstride: frequency of sampling from the data
		label: sequence for which actions. Only supports a single label at the moment
		min_cutoff: discard sequences with length below this threshold
		max_size: the maximum size of each sequence. If set to 0, it is calculated automatically
		based on the average length of the sequences.
		seq_gen_option: options are, 'trunc' truncates long sequences to the maximum size, 'divide' breaks the
		long sequences to smaller sizes no longer than maximum size, 'random' randomly (with a predefined seed) selects
		a subset of each sequence equal to the max_size 'none' returns the original sequences
		seq_overlap_rate: if 'divide' is selected, this defines how much divided sequences can overlap
		occlusion_removal_percent: threshold to discard sequences that contain occlusion.
		occlusion_removal_type: 'absolute', percentage calculated with respect to max size,
		'relative' percentage calculated based on average length of the sequences.
		balance: if true, it balances the number of positive and negative samples,
		optical_window: the length of images to for optical flow.
		use_fixed_stride: if set true, uses the 'fstride' defined, otherwise it selects a fstride calculated by
		dividing the size of the video sequence by the max_size number '''

		params = {'fstride': 30, 'label': "walking",
		'min_cutoff': 3, 'max_size': 10, 'seq_gen_option':'trunc',
		'occlusion_removal_percent':0.2,'seq_overlap_rate': 0.50,
		'occlusion_removal_type':'absolute','balance': True,
		'optical_flow_window': 10,'use_fixed_stride': True}
		for i in opts.keys():
			params[i] = opts[i]

		print('---------------------------------------------------------')
		print("Generating action sequence data")
		print(params)

		assert(params['seq_gen_option'] in ['none', 'trunc','divide','random'], \
			'Wrong sequence generation option is selected.\n Options: none, trunc, divide, random')

		self._image_set = image_set

		activities = self.generate_database_activity(self._image_set,frame_skip = 1)

		if params['label'] == 'crossing':
			sequence_data,frame_count = self._get_crossing_sequence_with_optical_flow(params, activities)
		elif params['label'] == 'intention':
			sequence_data,frame_count = self._get_trajectory_sequence_with_optical_flow(params, activities)
		else:
			sequence_data,frame_count = self._get_action_sequence_with_optical_flow(params, activities)

		adjusted_seq_data = self._adjust_seq_length(params, sequence_data,frame_count)
		no_occ_seq_data = self._remove_occluded_samples(params,adjusted_seq_data )
		balanced_seq_data= self._balance_samples_count(params,no_occ_seq_data)

		data_spec = 'jaad' \
					+'_skip_' + str(params['fstride']) \
					+'_lb_'  + params['label'] \
					+'_min_' + str(params['min_cutoff']) \
					+'_max_' + str(params['max_size']) \
					+'_seq_' + params['seq_gen_option'] \
					+'_occ_' + str(params['occlusion_removal_percent']) \
					+'_occ_type_' + params['occlusion_removal_type'] \
					+'_bal_' + str(params['balance'])

		if params['label'] == 'crossing':
			data_spec = data_spec + '_stride_fixed_' + str(params['use_fixed_stride'])
		print('---------------------------------------------------------')
		print("The %s sequence list is created!" %image_set)
		return {'imgPath':balanced_seq_data['gt_data'],
				'bbox':balanced_seq_data['bbox'],
				'labels':balanced_seq_data['gt_labels'],
				'optFlow_imgPath':balanced_seq_data['optical_flow'],
				'optFlow_bbox':balanced_seq_data['optical_flow_bbox'],
				'max_seq_length':params['max_size'],
				'data_spec': data_spec}
	def _get_trajectory_sequence_with_optical_flow_crossing_only(self,params, activities):
		'''
		Generates trajectory data. Each data point (frame) has an associated optical flow element
		which contains a seuqence of images (and bounding boxes) that include frames before and after
		(specified by optical flow window) the current frame.
		'''
		print('---------------------------------------------------------')
		print("Generating data sequences for crossing")

		# TODO: a moving window approach
		# a reference frame for before and after
		num_pedestrians = 0
		optical_window = 2 if params['optical_flow_window'] < 2 else params['optical_flow_window']
		seq_stride = params['fstride']
		# select the number of frames before and after a sequence. e.g. consider the current frame T.
		# if optical flow is an odd number such as 3 the cut out would be [T-1,T,T+1] and
		# if the optical window is an even number, e.g. 4, the cut out would be [T-2,T-1, T, T + 1]
		num_frames_before = optical_window//2
		num_frames_after = num_frames_before - 1 + optical_window % 2
		gt_data = []
		bbox = []
		gt_labels = []
		optical_flow = []
		optical_flow_bbox = []
		frame_count = []
		# Generates image sequences and corresponding optical flow windows.
		#  It adds an image to the sequence until the tag for that
		# label changes, e.g. from walking to standing. At this point the sequence is added to the
		#vecotr and a new sequence is formed.
		for vid in sorted(activities.keys()):
			for pid in sorted(activities[vid].keys()):
				if not activities[vid][pid]['images']:
					continue
				if params['crossing_frame_ref'] and (activities[vid][pid]['decision_frame'] == -1 \
				or activities[vid][pid]['is_crossing'] == -1):
					continue
				# Check if the pedestrian is relevant
				dec_frame = activities[vid][pid]['decision_frame']
				try:
					frame_index = activities[vid][pid]['frames'].index(dec_frame)
				except:
					print("Decision frame for %s in %s is out of bound due to scale limitation." %(pid,vid))
					continue

				is_crossing = activities[vid][pid]['is_crossing']
				decision_frame = activities[vid][pid]['decision_frame']

				print(is_crossing)
				sys.exit(1)
				seq = activities[vid][pid]['images']
				box = activities[vid][pid]['bboxes']
				trunc_seq = seq[num_frames_before:frame_index-num_frames_after]

				if len(trunc_seq) <= params['min_cutoff']:
					continue
				trunc_bbox = box[num_frames_before:frame_index-num_frames_after]
				opt_flow = [seq[i:i + optical_window] for i in range(len(seq)- optical_window)]
				opt_flow_b = [box[i:i + optical_window] for i in range(len(box)- optical_window)]
				if not stride_mode:
					seq_stride = 1 if len(trunc_seq) < params['max_size'] else len(trunc_seq)//params['max_size']
				num_pedestrians += 1
				gt_data.append(trunc_seq[::seq_stride])
				bbox.append(trunc_bbox[::seq_stride])
				frame_count.append(len(trunc_seq[::seq_stride]))
				optical_flow.append(opt_flow[::seq_stride])
				optical_flow_bbox.append(opt_flow_b[::seq_stride])
				gt_labels.append(activities[vid][pid]['is_crossing'])


		num_pos_samples = np.count_nonzero(np.array(gt_labels))
		print('Number of pedestrians: %d ' % num_pedestrians )
		print('Subset: %s'% self._image_set)
		print('Sample Count: Positive: %d  Negative: %d '
		% (num_pos_samples, len(gt_labels)-num_pos_samples))
		print('Sequence length: Min %d, Max %d, Avg. %0.2f\n' % (min(frame_count),
		 max(frame_count), sum(frame_count)/len(frame_count)))
		return {'gt_data': gt_data,
				'bbox': bbox,
				'optical_flow':optical_flow,
				'optical_flow_bbox':optical_flow_bbox,
				'gt_labels': gt_labels},frame_count

	def generate_data_trajectory_sequence(self,image_set,**opts):
		''' Generates trajectory seuqences:
		fstride: frequency of sampling from the data
		label: sequence for which actions. Only supports a single label at the moment
		min_cutoff: discard sequences with length below this threshold
		max_size: the maximum size of each sequence. If set to 0, it is calculated automatically
		based on the average length of the sequences.
		seq_overlap_rate: if 'divide' is selected, this defines how much divided sequences can overlap
		occlusion_removal_percent: threshold to discard sequences that contain occlusion.
		occlusion_removal_type: 'absolute', percentage calculated with respect to max size,
		balance: if true, it balances the number of positive and negative samples,
		optical_window: the length of images to for optical flow. '''

		params = {'fstride': 10,
		'max_size_observe': 10,
		'max_size_predict': 10,
		'occlusion_removal_percent':0.2,
		'seq_overlap_rate': 0.50,
		'occlusion_removal_type':'absolute',
		'crossing_frame_ref': False,
		'balance': True,
		'optical_flow_window': 2,
		'all_samples': False}

		for i in opts.keys():
			params[i] = opts[i]

		print('---------------------------------------------------------')
		print("Generating action sequence data")
		print(params)

		self._image_set = image_set
		if params['all_samples']:
			annotations = self.generate_database_trajectory_all(self._image_set,frame_skip = 1)
		else:
			annotations = self.generate_database_activity(self._image_set,frame_skip = 1)

		if params['seq_type'] == 'split':
			sequence_data = self._get_trajectory_sequence_with_optical_flow(params,annotations)
		elif params['seq_type'] == 'single':
			sequence_data = self._get_trajectory_sequence_with_optical_flow_single(params,annotations)


		# data_spec = 'jaad' \
		# 			+'_skip_' + str(params['fstride']) \
		# 			+'_lb_'  + params['label'] \
		# 			+'_min_' + str(params['min_cutoff']) \
		# 			+'_max_' + str(params['max_size']) \
		# 			+'_seq_' + params['seq_gen_option'] \
		# 			+'_occ_' + str(params['occlusion_removal_percent']) \
		# 			+'_occ_type_' + params['occlusion_removal_type'] \
		# 			+'_bal_' + str(params['balance'])
		# if params['label'] == 'crossing':
		# 	data_spec = data_spec + '_stride_fixed_' + str(params['use_fixed_stride'])
		# print('---------------------------------------------------------')
		# print("The %s sequence list is created!" %image_set)
		# return {'imgPath':balanced_seq_data['gt_data'],
		# 		'bbox':balanced_seq_data['bbox'],
		# 		'labels':balanced_seq_data['gt_labels'],
		# 		'optFlow_imgPath':balanced_seq_data['optical_flow'],
		# 		'optFlow_bbox':balanced_seq_data['optical_flow_bbox'],
		# 		'max_seq_length':params['max_size'],
		# 		'data_spec': data_spec}

		return sequence_data
	def generate_database_activity(self,image_set, data_size = (224,224),
	 just_ground_truth = True,frame_skip = 1):
		''' Generates and save a database of activities for all pedestriasns'''
		print('---------------------------------------------------------')
		print("Generating activity database for jaad %s"% image_set)
		self._image_set = image_set
		# Generates a list of behavioral xml file names for  videos
		cache_file = join(self.cache_path, self._name +'_'+ self._image_set +
		'_pedestrian_activities' + '_skip' + str(frame_skip) + '.pkl')
		if os.path.exists(cache_file) and not self._regen_pkl:
			with open(cache_file, 'rb') as fid:
				try:
					video_activities = pickle.load(fid)
				except:
					video_activities = pickle.load(fid, encoding='bytes')
			print('{} pedestrian activities loaded from {}'.format(self._name, cache_file))
			return video_activities
		self._video_ids = self._load_video_ids()
		ped_decision_points = self._get_ped_decision_point()

		video_activities ={}
		for vid in self._video_ids:
			#gets all pedestrian behaviors for each video
			# ped_behaviours = self._get_pedestrian_activities_video(id)
			image_index = self._load_image_set_index_from_video_id(vid,frame_skip)
			pedestrians = self._initialize_dict_behavior(vid)
			for imgIdx in image_index:
				box = self._load_annotation(imgIdx)
				# print(imgIdx)
				if not box['boxes'].any():
					continue
				# read images full path from indecies
				imPath = self.image_path_from_index(imgIdx)
				# get bounding boxes for each pedestrian
				ped_bbox = self._get_pedestrian_bbox(box)
				# load activities of pedestrians for the given image
				ped_acts = self._get_pedestrian_activities_image(imgIdx)
				# convert activities to groundtruth vector
				ped_gt_labels = self._generate_labels_for_pedestrian_activities(ped_acts)
				# load activities of driver for the given image
				driver_act = self._get_driver_activities_image(imgIdx)
				# convert activities to groundtruth vector
				dr_gt_labels = self._generate_labels_for_driver_activities(driver_act)
				cropped_sample = {}
				if not just_ground_truth:
					cropped_sample = self._crop_samples(ped_bbox,data_size,imPath)
				for ped in sorted(ped_acts.keys()):
					if ped not in sorted(ped_bbox.keys()):
						continue
					pedestrians[ped]['images'].append(imPath)
					pedestrians[ped]['bboxes'].append(ped_bbox[ped])
					pedestrians[ped]['actions'].append(ped_acts[ped])
					pedestrians[ped]['actions_gts'].append(ped_gt_labels[ped])
					pedestrians[ped]['driver_actions'].append(driver_act)
					pedestrians[ped]['driver_gts'].append(dr_gt_labels)
					pedestrians[ped]['frames'].append(self._get_frame_number(imgIdx))
					if not just_ground_truth:
						pedestrians[ped]['samples'].append(cropped_sample[ped])
			for ped in pedestrians:
				pedestrians[ped]['is_crossing'] = ped_decision_points[vid][ped]['is_crossing']
				pedestrians[ped]['decision_frame'] = ped_decision_points[vid][ped]['decision_frame']

			video_activities[vid] = pedestrians
		with open(cache_file, 'wb') as fid:
			pickle.dump(video_activities, fid, pickle.HIGHEST_PROTOCOL)
		print('wrote activites to {}'.format(cache_file))
		return video_activities
	def _get_pedestrian_bbox_all(self,box):
		'''Returns the bounding box of pedestrians'''
		bbox = {}
		for b,t,o in zip(box['boxes'],box['tags'], box['occlusion']):
			# Check whether there is a pedestrian tags on bounding boxes
			if t.find('ped') != -1:
				# Split is for the ones that have
				# separated ids, e.g. pedestrians_p1 and pedestrian_p2
				ped_id = t.split("_")[0]
				bbox[ped_id] = np.append(b,o)
		return bbox

	def generate_database_trajectory_all(self,image_set,frame_skip = 1):
		''' Generates and save a database of trajectories for all annotated pedestriasns'''
		print('---------------------------------------------------------')
		print("Generating trajectories database for jaad %s all pedestrians"% image_set)
		self._image_set = image_set
		# Generates a list of behavioral xml file names for  videos
		cache_file = join(self.cache_path, self._name +'_'+ self._image_set +
		'_pedestrian_trajectories_all' + '_skip' + str(frame_skip) + '.pkl')
		if os.path.exists(cache_file) and not self._regen_pkl:
			with open(cache_file, 'rb') as fid:
				try:
					video_trajectories = pickle.load(fid)
				except:
					video_trajectories = pickle.load(fid, encoding='bytes')
			print('{} pedestrian trajectories loaded from {}'.format(self._name, cache_file))
			return video_trajectories
		self._video_ids = self._load_video_ids()
		ped_decision_points = self._get_ped_decision_point()

		video_trajectories ={}
		for vid in self._video_ids:
			#gets all pedestrian behaviors for each video
			# ped_behaviours = self._get_pedestrian_activities_video(id)
			image_index = self._load_image_set_index_from_video_id(vid,frame_skip)
			pedestrians = {}
			for imgIdx in image_index:
				box = self._load_annotation(imgIdx)
				if not box['boxes'].any():
					continue
				# read images full path from indecies
				imPath = self.image_path_from_index(imgIdx)
				# get bounding boxes for each pedestrian
				ped_bbox = self._get_pedestrian_bbox_all(box)

				for ped in sorted(ped_bbox.keys()):
					try:
						pedestrians[ped]['images'].append(imPath)
					except:
						pedestrians[ped] = {'images':[],'bboxes':[],'frames':[]}
						pedestrians[ped]['images'].append(imPath)
					pedestrians[ped]['bboxes'].append(ped_bbox[ped])
					pedestrians[ped]['frames'].append(self._get_frame_number(imgIdx))

			video_trajectories[vid] = pedestrians
		with open(cache_file, 'wb') as fid:
			pickle.dump(video_trajectories, fid, pickle.HIGHEST_PROTOCOL)
		print('wrote trajectories to {}'.format(cache_file))
		return video_trajectories
	def _get_trajectory_sequence_with_optical_flow(self,params, activities):
		'''
		Generates trajectory data. Each data point (frame) has an associated optical flow element
		which contains a seuqence of images (and bounding boxes) that include frames before and after
		(specified by optical flow window) the current frame.
		'''
		print('---------------------------------------------------------')
		print("Generating trajectory data")

		# TODO: a moving window approach
		# 		a reference frame for before and after
		num_pedestrians = 0
		optical_window = 2 if params['optical_flow_window'] < 2 else params['optical_flow_window']
		seq_stride = params['fstride']
		max_obs_size = params['max_size_observe']
		max_pred_size = params['max_size_predict']

		# select the number of frames before and after a sequence. e.g. consider the current frame T.
		# if optical flow is an odd number such as 3 the cut out would be [T-1,T,T+1] and
		# if the optical window is an even number, e.g. 4, the cut out would be [T-2,T-1, T, T + 1]
		num_frames_before = optical_window//2
		num_frames_after = num_frames_before - 1 + optical_window % 2

		image_observe = []
		bbox_observe = []
		center_observe = []
		image_predict = []
		bbox_predict = []
		center_predict = []

		intention_label = []

		optflow_observe = []
		bbox_optflow_observe = []
		optflow_predict = []
		bbox_optflow_predict = []

		frame_count = []
		overlap_stride = max_obs_size if params['seq_overlap_rate'] == 0 else \
		int((1 - params['seq_overlap_rate'])*max_obs_size)
		overlap_stride = 1 if overlap_stride < 1 else overlap_stride


		max_total = max_obs_size + max_pred_size
		# Generates image sequences and corresponding optical flow windows.
		#  It adds an image to the sequence until the tag for that
		# label changes, e.g. from walking to standing. At this point the sequence is added to the
		#vecotr and a new sequence is formed.
		for vid in sorted(activities.keys()):
			for pid in sorted(activities[vid].keys()):
				if not activities[vid][pid]['images']:
					continue
				seq = activities[vid][pid]['images']
				box = activities[vid][pid]['bboxes']
				box = [b[0:4] for b in box]

				center = [[(b[0]+b[2])/2,(b[1]+b[3])/2] for b in box]

				# adjust the number of sample points according to optical flow window size
				trunc_seq = seq[num_frames_before:len(seq)-num_frames_after]
				trunc_seq = trunc_seq[::seq_stride]
				if len(trunc_seq) < max_obs_size + max_pred_size :
					continue
				trunc_box = box[num_frames_before:len(box)-num_frames_after]
				trunc_box = trunc_box[::seq_stride]

				trunc_center = center[num_frames_before:len(center)-num_frames_after]
				trunc_center = trunc_center[::seq_stride]

				opt_flow = [seq[i:i + optical_window] for i in range(len(seq)- optical_window)]
				opt_flow_b = [box[i:i + optical_window] for i in range(len(box)- optical_window)]

				opt_flow = opt_flow[::seq_stride]
				opt_flow_b = opt_flow_b[::seq_stride]

				num_pedestrians += 1

				# Generate data samples given the overlap
				image_obs = [trunc_seq[x : x + max_obs_size] for x in range(0,len(trunc_seq) - max_total+1, overlap_stride)]
				bbox_obs = [trunc_box[x : x + max_obs_size] for x in range(0,len(trunc_box) - max_total+1, overlap_stride)]
				center_obs = [trunc_center[x : x + max_obs_size] for x in range(0,len(trunc_center) - max_total+1, overlap_stride)]

				image_pred = [trunc_seq[max_obs_size + x : max_obs_size + x + max_pred_size] for x in range(0,len(trunc_seq) - max_total+1, overlap_stride)]
				bbox_pred = [trunc_box[max_obs_size + x : max_obs_size + x + max_pred_size] for x in range(0,len(trunc_box) - max_total+1, overlap_stride)]
				center_pred = [trunc_center[max_obs_size + x : max_obs_size + x + max_pred_size] for x in range(0,len(trunc_center) - max_total+1, overlap_stride)]

				# Sanity check
				for io,bo,ip,bp in zip(image_obs,bbox_obs,image_pred, bbox_pred):
					if len(bp) != len(ip):
						print('numbers don t match')
				# for i in range(0,len(image_obs)):
				# 	visualize_trajectory_predict_cv(image_obs[i],bbox_obs[i], window_name = 'obs')
				# 	visualize_trajectory_predict_cv(image_pred[i],bbox_pred[i], window_name = 'pred')

				opf_obs = [opt_flow[x : x + max_obs_size] for x in range(0,len(opt_flow) - max_total+1, overlap_stride)]
				bbox_opf_obs = [opt_flow_b[x : x + max_obs_size] for x in range(0,len(opt_flow_b) - max_total+1, overlap_stride)]
				opf_pred = [opt_flow[max_obs_size + x : max_obs_size + x + max_pred_size] for x in range(0,len(opt_flow) - max_total+1, overlap_stride)]
				bbox_opf_pred = [opt_flow_b[max_obs_size + x : max_obs_size + x + max_pred_size] for x in range(0,len(opt_flow_b) - max_total+1, overlap_stride)]

				image_observe.extend(image_obs)
				bbox_observe.extend(bbox_obs)
				center_observe.extend(center_obs)

				image_predict.extend(image_pred)
				bbox_predict.extend(bbox_pred)
				center_predict.extend(center_pred)

				optflow_observe.extend(opf_obs)
				bbox_optflow_observe.extend(bbox_opf_obs)
				optflow_predict.extend(opf_pred)
				bbox_optflow_predict.extend(bbox_opf_pred)

		print('Number of pedestrians: %d ' % num_pedestrians )
		print('Subset: %s'% self._image_set)
		print('Total number of samples: %d '% len(image_observe))

		return {'image_observe': image_observe,
				'bbox_observe': bbox_observe,
				'center_observe':center_observe,
				'image_predict': image_predict,
				'bbox_predict': bbox_predict,
				'center_predict':center_predict,
				'optical_flow_observe':optflow_observe,
				'optical_flow_bbox_observe':bbox_optflow_observe,
				'optical_flow_predict':optflow_predict,
				'optical_flow_bbox_predict':bbox_optflow_predict,
				'intention_label': intention_label}
	def _get_trajectory_sequence_with_optical_flow_single(self, params, annotations):
		'''
		Generates trajectory data. Each data point (frame) has an associated optical flow element
		which contains a seuqence of images (and bounding boxes) that include a number of frames
		(specified by optical flow window) before and after the current frame.
		This function just returns all samples as a single vector for each data type
		'''
		print('---------------------------------------------------------')
		print("Generating trajectory data")

		# TODO: a reference frame for before and after
		num_pedestrians = 0
		optical_window = 2 if params['optical_flow_window'] < 2 else params['optical_flow_window']
		seq_stride = params['fstride']
		max_obs_size = params['max_size_observe']
		max_pred_size = params['max_size_predict']

		image_seq = []
		box_seq = []
		center_seq = []
		intent_seq = []
		opt_flow_seq = []
		opt_flow_box = []
		pids_seq = []
			# select the number of frames before and after a sequence. e.g. consider the current frame T.
		# if optical flow is an odd number such as 3 the cut out would be [T-1,T,T+1] and
		# if the optical window is an even number, e.g. 4, the cut out would be [T-2,T-1, T, T + 1]
		num_frames_before = optical_window//2
		num_frames_after = num_frames_before - 1 + optical_window % 2
		max_total = max_obs_size + max_pred_size

		# Generates image sequences and corresponding optical flow windows.
		#  It adds an image to the sequence until the tag for that
		# label changes, e.g. from walking to standing. At this point the sequence is added to the
		#vecotr and a new sequence is formed.
		for vid in sorted(annotations.keys()):
			for pid in sorted(annotations[vid].keys()):
				ped_id = vid + '_' + pid
				if not annotations[vid][pid]['images']:
					continue
				seq = annotations[vid][pid]['images']
				box = annotations[vid][pid]['bboxes']
				box = [b[0:4] for b in box]
				ped_ids = [[ped_id]]*len(box)
				if not params['all_samples']:
					if annotations[vid][pid]['is_crossing'] == -1:
						intent = [[0]]*len(box)
					else:
						intent = [[1]]*len(box)
				else:
					intent = [[0]]*len(box)
				center = [[(b[0]+b[2])/2,(b[1]+b[3])/2] for b in box]

				if len(seq[num_frames_before:len(seq) - num_frames_after])/seq_stride < max_total :
					continue


				image_seq.append(self._truncate_and_sample_sequence(seq, num_frames_before,num_frames_after,seq_stride))
				box_seq.append(self._truncate_and_sample_sequence(box, num_frames_before,num_frames_after,seq_stride))
				center_seq.append(self._truncate_and_sample_sequence(center, num_frames_before,num_frames_after,seq_stride))
				intent_seq.append(self._truncate_and_sample_sequence(intent, num_frames_before,num_frames_after,seq_stride))
				pids_seq.append(self._truncate_and_sample_sequence(ped_ids, num_frames_before,num_frames_after,seq_stride))
				num_pedestrians += 1

				# Optical flow frames
				opt_flow = [seq[i:i + optical_window] for i in range(len(seq)- optical_window)]
				opt_flow_b = [box[i:i + optical_window] for i in range(len(box)- optical_window)]
				opt_flow_seq.extend(opt_flow[::seq_stride])
				opt_flow_box.extend(opt_flow_b[::seq_stride])

		print('Number of pedestrians: %d ' % num_pedestrians)
		print('Subset: %s'% self._image_set)
		print('Total number of samples: %d '% len(image_seq))

		return {'image': image_seq,
				'pid': pids_seq,
				'bbox': box_seq,
				'center': center_seq,
				'intent': intent_seq, #intent_seq,
				'optical_flow':opt_flow_seq,
				'optical_flow_bbox':opt_flow_box,
				'data_info':{'observe_length':  params['max_size_observe'],
							'predict_length': params['max_size_predict']}}
	def _truncate_and_sample_sequence(self,seq, nf_before,nf_after,sample_stride):
		trunc_seq = seq[nf_after:len(seq) - nf_after]
		return trunc_seq[::sample_stride]
if __name__ == '__main__':
  from jaad.jaad_data import jaad

  d = jaad('train')
  res = d.roidb
  from IPython import embed;

  embed()
