from data_fixed import Data
import numpy as np

data = Data.get_data_paths("./calving", "./rcnn", percentage_a=1,  percentage_b=1)
np_data = np.array(data)
np.random.shuffle(np_data)
print(len(np_data))
np.save("prepared_ids", np_data)
