import sys, os, shutil
import argparse
import bob
from features import mfcc
import numpy as np
import cPickle
import scipy.sparse
import scipy.io.wavfile as wav


#mfcc_dir_path = '/mnt/alderaan/mlteam3/Assignment4/data/voxforge_de_mfcc_39D'
mfcc_dir_path = '/mnt/alderaan/mlteam3/Assignment4/data/vox_for_training_model_balanced30fold/'
output_mfcc_path = '/mnt/alderaan/mlteam3/Assignment4/data/vox_for_training_model_balanced30fold_concat/'


if not os.path.exists(output_mfcc_path):
    os.makedirs(output_mfcc_path)

mfcc_dimensions = 39

def getMfccVector(noise_mix_speech_file):
    (rate, signal) = wav.read(noise_mix_speech_file)
    mfcc_vec = mfcc(signal,rate,winlen=0.025,winstep=0.01,numcep=mfcc_dimensions,
          nfilt=mfcc_dimensions*2,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
          ceplifter=22,appendEnergy=True)
    return mfcc_vec

def unpackMfccVector(noise_mix_speech_file):
    with open(noise_mix_speech_file, 'rb') as infile1:
        mfcc = cPickle.load(infile1)
    infile1.close()
    mfcc = scipy.sparse.coo_matrix((mfcc), dtype=np.float64).toarray()
    return mfcc

def saveVectorToDisk(mfcc_vector_output_file, speech_vector_final):
    mfcc_vector_file = open(mfcc_vector_output_file, 'w')
    temp1 = scipy.sparse.coo_matrix(speech_vector_final)
    cPickle.dump(temp1,mfcc_vector_file,-1)
    mfcc_vector_file.close()


def concatenateSpeakerWise(mfcc_dir_path):
    for root, dirs, files in os.walk(mfcc_dir_path):

        path = root.split('/')
        speech_vector_final = np.zeros((1,mfcc_dimensions))
        speech_vector_final = np.delete(speech_vector_final, (0), axis=0)

        mfcc_vector_output_file=''
        for file in files:
            if (file.lower().endswith('.dat')):
                #print "Current File: " + str(file)
                #sys.exit()
                #name  = file.replace('.dat','')

                speechFilePath = os.path.join(root,str(file))
                tmp = os.path.dirname(speechFilePath)
                root_dir_name = os.path.basename(tmp)

                mfcc_file_path = os.path.join(mfcc_dir_path,str(root_dir_name),file)
                mfcc_vector =  unpackMfccVector(mfcc_file_path)
                speech_vector_final = np.vstack((speech_vector_final,mfcc_vector))

                mfcc_vector_output_file = os.path.join(output_mfcc_path, str(root_dir_name)+'_mfcc.dat')
                #print str(mfcc_vector_output_file)
        print 'Path: '+str(mfcc_vector_output_file)
        if (mfcc_vector_output_file!=''):
            saveVectorToDisk(mfcc_vector_output_file, speech_vector_final)
            #sys.exit()


def main():
    concatenateSpeakerWise(mfcc_dir_path)
    print "Finished!"


if __name__=="__main__":
    main()
