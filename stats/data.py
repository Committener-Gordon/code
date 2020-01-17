import os
import random

class Data:
    def get_data_paths(set_a, set_b, percentage_a=1, percentage_b=1):
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(set_a):
            for file in f:
                if '.mp4' in file and random.random() < percentage_a :
                    files.append(os.path.join(r, file))

        for r, d, f in os.walk(set_b):
            for file in f:
                if '.mp4' in file and random.random() < percentage_b :
                    files.append(os.path.join(r, file))        
        return files
