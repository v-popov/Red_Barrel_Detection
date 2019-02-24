import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
from roipoly import RoiPoly

train_data_folder = 'Train_Set'

# Select one of the following folders depending on what are you going to select:

#labeled_data_folder = 'labeled_data_red'
#labeled_data_folder = 'labeled_data_not_red'
#labeled_data_folder = 'labeled_data_almost_red'
labeled_data_folder = 'barrels'

def select_region(filename):
    gray = cv2.imread(train_data_folder + '/' + filename,0)
    img = cv2.imread(train_data_folder + '/' + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig = plt.figure()
    plt.imshow(img)

    roi1 = RoiPoly(color='r', fig=fig)
    region = roi1.get_mask(gray)

    name = filename.split('.png')[0]
    np.save(labeled_data_folder + '/' + name, region)
 
if __name__ == '__main__':
    files = os.listdir(train_data_folder)
    for filename in files:
        select_region(filename)



    


