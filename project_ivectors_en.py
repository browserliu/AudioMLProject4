import os
import bob
import numpy
import shutil
import cPickle
import scipy.sparse
import facereclib.utils as utils
from sklearn import preprocessing

# parameters
training_threshold = 5e-4
variance_threshold = 5e-4
max_iterations = 25
relevance_factor = 4             # Relevance factor as described in Reynolds paper
gmm_enroll_iterations = 1        # Number of iterations for the enrollment phase
subspace_dimension_of_t = 100    # This is the dimension of the T matrix.
INIT_SEED = 5489

# parameters for the GMM
gaussians = "512"

ubm_hdf5 = "/mnt/alderaan/mlteam3/Assignment4/data/gmm_ubm_en_39D/gmm_giantubm_512G.hdf5"
tv_hdf5 = 'model/tv_512G_EN.hdf5'
input_ubm_features = '/mnt/alderaan/mlteam3/Assignment4/data/vox_for_ubm'
input_speaker_train_features = '/mnt/alderaan/mlteam3/Assignment4/data/vox_for_training_model_balanced30fold'
input_speaker_test_features = '/mnt/alderaan/mlteam3/Assignment4/data/vox_for_training_probe_balanced30fold'

def normalize(data):
    return preprocessing.normalize(data,norm='l2')

def read_mfcc_features(input_features):

    with open(input_features, 'rb') as file:
        features = cPickle.load(file)
    features = scipy.sparse.coo_matrix((features),dtype=numpy.float64).toarray()
    if features.shape[1] != 0:
        features = normalize(features)
    return features

def load_training_gmmstats(input_features):

    gmm_stats_list = []
    for root, dir, files in os.walk(input_features):
        for file in files:
            features_path = os.path.join(root, str(file))
            features = read_mfcc_features(features_path)
            stats = bob.machine.GMMStats(ubm.dim_c, ubm.dim_d)
            ubm.acc_statistics(features, stats)
            gmm_stats_list.append(stats)

    return gmm_stats_list

def train_enroller(input_features):

    # load GMM stats from UBM training files 
    gmm_stats = load_training_gmmstats(input_features)  

    # Training IVector enroller
    print "training enroller (total variability matrix) ", max_iterations, 'max_iterations'
    # Perform IVector initialization with the UBM
    ivector_machine = bob.machine.IVectorMachine(ubm, subspace_dimension_of_t) 
    ivector_machine.variance_threshold = variance_threshold

    # Creates the IVectorTrainer and trains the ivector machine
    ivector_trainer = bob.trainer.IVectorTrainer(update_sigma=True, convergence_threshold=variance_threshold, max_iterations=max_iterations)
    # An trainer to extract i-vector (i.e. for training the Total Variability matrix)
    ivector_trainer.train(ivector_machine, gmm_stats)
    ivector_machine.save(bob.io.HDF5File(tv_hdf5, 'w'))
    print "IVector training: saved enroller's IVector machine base to '%s'" % tv_hdf5

    return ivector_machine

def lnorm_ivector(ivector):
    norm = numpy.linalg.norm(ivector)
    if norm != 0:
        return ivector/numpy.linalg.norm(ivector)
    else:
        return ivector

def save_ivectors(data, feature_file):
    hdf5file = bob.io.HDF5File(feature_file, "w")
    hdf5file.set('ivec', data)

def project_ivectors(input_features):
    """Extract the ivectors for all files of the database"""
    print "projecting ivetors"
    tv_enroller = bob.machine.IVectorMachine(ubm, subspace_dimension_of_t)
    tv_enroller.load(bob.io.HDF5File(tv_hdf5))
    #print input_features
    for root, dir, files in os.walk(input_features):
        ivectors = []
        for file in files:
            features_path = os.path.join(root, str(file))
            features = read_mfcc_features(features_path)
            stats = bob.machine.GMMStats(ubm.dim_c, ubm.dim_d)
            ubm.acc_statistics(features, stats)
            ivector = tv_enroller.forward(stats)   
            lnorm_ivector(ivector)
            ivectors.append(ivector)

        ivectors_path = 'ivectors/ivectors_vox_ENubm_ubm/'+ str(gaussians) + '/' + input_features.split('/')[-1]
        if not os.path.exists(ivectors_path):
            os.makedirs(ivectors_path)
        ivectors_path = ivectors_path + '/' + os.path.split(root)[1] + '_' + str(gaussians) + '.ivec'
        save_ivectors(ivectors, ivectors_path)
        print "saved ivetors to '%s' " % ivectors_path

#############################################

ubm = bob.machine.GMMMachine(bob.io.HDF5File(ubm_hdf5))
#train_enroller(input_ubm_features)
project_ivectors(input_ubm_features)
#project_ivectors(input_speaker_train_features)
#project_ivectors(input_speaker_test_features)


