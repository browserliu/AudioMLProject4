import numpy as np
import shutil
import cPickle
import scipy.sparse
import random
import os
import sys

vox_ubm = "/mnt/alderaan/mlteam3/Assignment4/data/OLD_vox_80%_speaker_level"
vox_speaker_model = "/mnt/alderaan/mlteam3/Assignment4/data/OLD_vox_20%_speaker_level"
vox_all = "/mnt/alderaan/mlteam3/Assignment4/data/ALL_vox_speaker_vaded"

file_path = vox_speaker_model

split_fold = 30
threshold = 5000

for root, dir, files in os.walk(file_path):
	for file in files:
		mfcc_file = os.path.join(root, str(file))
		with open(mfcc_file, 'rb') as infile1:
			mfcc = cPickle.load(infile1)
		infile1.close()
		mfcc = scipy.sparse.coo_matrix((mfcc), dtype=np.float64).toarray()
		if mfcc.shape[0] > threshold:
			mfcc = mfcc[:5000]
			fold_num = mfcc.shape[0] / split_fold
			for x in xrange(0, split_fold):
				if x < 18: 
					mfcc_split = mfcc[x*fold_num:(x+1)*fold_num]
					mfcc_split_path= "/mnt/alderaan/mlteam3/Assignment4/data/vox_for_training_model_balanced30fold/" + file[:-9]
					if not os.path.exists(mfcc_split_path):
						os.makedirs(mfcc_split_path)
					mfcc_split_filename = mfcc_split_path + "/" + file[:-9] + "_" + str(x) + '.dat'
					mfcc_file = open(mfcc_split_filename, 'w')
					temp1 = scipy.sparse.coo_matrix(mfcc_split)
					cPickle.dump(temp1,mfcc_file,-1)
					mfcc_file.close()
				else: 
					mfcc_split = mfcc[x*fold_num:(x+1)*fold_num]
					mfcc_split_path= "/mnt/alderaan/mlteam3/Assignment4/data/vox_for_training_probe_balanced30fold/" + file[:-9]
					if not os.path.exists(mfcc_split_path):
						os.makedirs(mfcc_split_path)
					mfcc_split_filename = mfcc_split_path + "/" + file[:-9] + "_" + str(x) + '.dat'
					mfcc_file = open(mfcc_split_filename, 'w')
					temp1 = scipy.sparse.coo_matrix(mfcc_split)
					cPickle.dump(temp1,mfcc_file,-1)
					mfcc_file.close()