import os
import bob
import numpy
import math
import bob.blitz
from itertools import *
import pandas
import machine_training

whitening_enroler_file = 'model/whitening_ENubm_vox.hdf5'
wccn_enroller_file = 'model/wccn_ENubm_vox.hdf5'
lda_enroller_file = 'model/lda_ENubm_vox.hdf5'
pca_enroller_file = 'model/pca_ENubm_vox.hdf5'
plda_enroller_file = 'model/plda_ENubm_vox.hdf5'


# INPUT DIRECTORY
input_ubm_features = '/mnt/alderaan/mlteam3/Assignment4/data/vox_for_ubm'
train_ivectors_path = '/mnt/alderaan/mlteam3/Assignment4/ivectors/ivectors_vox_ENubm_balanced30fold/512/vox_for_training_probe_balanced30fold'
test_ivectors_path = '/mnt/alderaan/mlteam3/Assignment4/ivectors/ivectors_vox_ENubm_balanced30fold/512/vox_for_training_model_balanced30fold'

speaker_list = ["Steltek_512.ivec","stephsphynx_512.ivec","steviehs_512.ivec","Susi_512.ivec","theduke_512.ivec","TheLinuxist_512.ivec","thhoof_512.ivec","thisss_512.ivec","Thomas_512.ivec","timobaumann_512.ivec","tolleiv_512.ivec","UrbanCMC_512.ivec","vfsh_512.ivec","wasawasa_512.ivec","wmwie_512.ivec"]

def read_ivectors(ivector_file):
	ivectors_matrix = []
	for root, dir, files in os.walk(ivector_file):
	    for file in files:
	        ivec_path = os.path.join(root, str(file))
	        ivec = bob.io.HDF5File(ivec_path)
	        ivector = ivec.read('ivec')
	        ivector = numpy.array(ivector)
	        ivectors_matrix.append(ivector)

	return ivectors_matrix

def cosine_distance(a, b):
	if len(a) != len(b):
		raise ValueError, "a and b must be same length"
	numerator = sum(tup[0] * tup[1] for tup in izip(a,b))
	denoma = sum(avalue ** 2 for avalue in a)
	denomb = sum(bvalue ** 2 for bvalue in b)
	result = numerator / (numpy.sqrt(denoma)*numpy.sqrt(denomb))
	return result

def cosine_score(client_ivectors, probe_ivector):
	"""Computes the score for the given model and the given probe using the scoring function"""
	scores = []
	for ivec in client_ivectors:
		scores.append(cosine_distance(ivec, probe_ivector))
	return numpy.max(scores)



whitening_machine = bob.machine.LinearMachine(bob.io.HDF5File(whitening_enroler_file, 'r'))
lda_machine = bob.machine.LinearMachine(bob.io.HDF5File(lda_enroller_file, 'r'))
wccn_machine = bob.machine.LinearMachine(bob.io.HDF5File(wccn_enroller_file, 'r'))

accuracy_list = []
correct_total = 0.0
wrong_total = 0.0
for probe in speaker_list:
	correct = 0.0
	wrong = 0.0
	probe_ivectors_path = os.path.join(test_ivectors_path, probe)
	print 'Probe Speaker', probe
	probe_ivec = bob.io.HDF5File(probe_ivectors_path)
	probe_ivectors = probe_ivec.read('ivec')
	probe_ivectors = numpy.array(probe_ivectors)
	for probe_ivector in probe_ivectors:
		probe_result = []
		for model in speaker_list:
			model_ivec_path = os.path.join(train_ivectors_path, model)
			# whiten
			whitened_model_ivectors = machine_training.project_whitening_ivec(whitening_machine, model_ivec_path)
			whitened_probe_ivector = whitening_machine.forward(probe_ivector)
			normalized_probe_ivector = whitened_probe_ivector/numpy.linalg.norm(whitened_probe_ivector)

			score = cosine_score(whitened_model_ivectors, normalized_probe_ivector)
			probe_result += [(probe.split('.',1)[0], model.split('.',1)[0], score)]
		s_max = float("-Infinity")
		predict = []
		for item in probe_result:
			 p, m, s = item
			 if s > s_max:
			 	s_max = s
			 	predict = item
		print predict
		p, m, s = predict
		if p == m:
			correct = correct + 1
			correct_total += 1
		else:
			wrong = wrong + 1
			wrong_total += 1
	accuracy = correct / ( correct + wrong)
	accuracy_list += [(probe.split('.',1)[0], accuracy)]

print correct_total / ( correct_total + wrong_total)
df = pandas.DataFrame(accuracy_list)
df.columns = ["probe", "accuracy"]
df.to_csv("results/ivector_whiten_ENubm_balanced30fold.csv")