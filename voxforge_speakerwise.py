import numpy as np
import os 
import shutil

file_path = '/mnt/alderaan/mlteam3/voxforge_de/16kHz_16bit'
speaker_path = '/mnt/alderaan/mlteam3/voxforge_de/speakerwise'

for root, dir, files in os.walk(file_path):
        for file in files:
            if len(root.split('/')) == 8:
                if root.split('/')[7] == 'wav':
                    speaker_name = root.split('/')[6]
                    speaker_name = speaker_name.split('-')[0]
                    speaker_file = speaker_path + '/' + speaker_name 
                    if not os.path.exists(speaker_file):
                        os.makedirs(speaker_file)
                    source = os.path.join(root, file)
                    print source
                    destination = os.path.join(speaker_file, file)
                    print destination
                    shutil.move(source, destination)