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

def load_plda_enroller(plda_enroller_file):
	"""Reads the PCA projection matrix and the PLDA model from file"""
	# read UBM
	proj_hdf5file = bob.io.HDF5File(plda_enroller_file)
	proj_hdf5file.cd('/pca')
	pca_machine = bob.machine.LinearMachine(proj_hdf5file)
	proj_hdf5file.cd('/plda')
	plda_base = bob.machine.PLDABase(proj_hdf5file)
	plda_machine = bob.machine.PLDAMachine(plda_base)
	return pca_machine, plda_base

def perform_pca_client(pca_machine, client):
	"""Perform PCA on an array"""
	client_data_list = []
	for feature in client:
		# project data
		projected_feature = numpy.ndarray(pca_machine.shape[1], numpy.float64)
		projected_feature = pca_machine(feature)
		# add data in new array
		client_data_list.append(projected_feature)
	client_data = numpy.vstack(client_data_list)
	return client_data
	
def plda_enroll(pca_machine, plda_base, enroll_features):
	"""Enrolls the model by computing an average of the given input vectors"""
	#enroll_features = numpy.vstack(enroll_features)
	#enroll_features_projected = perform_pca_client(pca_machine, enroll_features)
	features =[ numpy.array([feature]) for feature in enroll_features]
	plda_trainer = bob.trainer.PLDATrainer()
	plda_trainer.train(plda_base, features)
	return plda_base

def read_plda_model(model_file):
	"""Reads the model, which in this case is a PLDA-Machine"""
	# read machine and attach base machine
	print ("model: %s" %model_file)
	plda_machine = bob.machine.PLDAMachine(bob.io.HDF5File(str(model_file)), plda_base)
	return plda_machine

def plda_score(model, probe):
	return model.compute_log_likelihood(probe)

whitening_machine = bob.machine.LinearMachine(bob.io.HDF5File(whitening_enroler_file, 'r'))
lda_machine = bob.machine.LinearMachine(bob.io.HDF5File(lda_enroller_file, 'r'))
wccn_machine = bob.machine.LinearMachine(bob.io.HDF5File(wccn_enroller_file, 'r'))

pca_machine, plda_base = load_plda_enroller(plda_enroller_file)

# plda
accuracy_list = []
for probe in speaker_list:
	correct = 0.0
	wrong = 0.0
	probe_ivectors_path = os.path.join(test_ivectors_path, probe)
	print 'Probe Speaker-->', probe
	probe_ivec = bob.io.HDF5File(probe_ivectors_path)
	probe_ivectors = probe_ivec.read('ivec')
	probe_ivectors = numpy.array(probe_ivectors)
	for probe_ivector in probe_ivectors:
		probe_result = []
		for model in speaker_list:
			model_ivec_path = os.path.join(train_ivectors_path, model)
			model_ivec = bob.io.HDF5File(model_ivec_path)
			model_ivectors = model_ivec.read('ivec')
			model_ivectors = numpy.array(model_ivectors)
			plda_base = plda_enroll(pca_machine, plda_base, model_ivectors)
			plda_machine = bob.machine.PLDAMachine(plda_base)
			score = plda_score(plda_machine, normalized_probe_ivector)
			probe_result += [(probe.split('.',1)[0], model.split('.',1)[0], score)]
		s_max = float("-Infinity") 
		predict = []
		print probe_result
		for item in probe_result:
			 p, m, s = item
			 if s > s_max:
			 	s_max = s
			 	predict = item
		print "predict:", predict
		if len(predict)>0:
			p, m, s = predict
			if p == m:
				correct = correct + 1
			else:
				wrong = wrong + 1
	accuracy = correct / ( correct + wrong)
	accuracy_list += [(probe.split('.',1)[0], accuracy)]

df = pandas.DataFrame(accuracy_list)
df.columns = ["probe", "accuracy"]
df.to_csv("results/ivector_plda_ENubm_balanced30fold.csv")