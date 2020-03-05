from data import Data
import numpy as np

data = Data.get_data_paths("./calving_samples", "./random_samples", percentage_a=0.1,  percentage_b=0.8)
np_data = np.array(data)
np.random.shuffle(np_data)
print(len(np_data))
np.save("prepared_ids", np_data)
