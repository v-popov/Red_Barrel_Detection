import os
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
import matplotlib.pyplot as plt

train_folder = 'Train_Set'
barrels_folder = 'barrels'

def get_train_pixels_coord(filename, pixels_fraction):
    pixels = np.load(filename)
    all_pix_size = len(pixels)
    
    pixels = np.argwhere(pixels==True)
    true_pix_size = len(pixels)

    np.random.seed(seed=1) # for reproducibility
    train_ind = np.random.randint(0, true_pix_size, size=round(true_pix_size * pixels_fraction))
    return pixels[train_ind], true_pix_size, all_pix_size


def create_train_file(folder, pixels_fraction):
    matrix = np.zeros((0,3))
    files = os.listdir(folder)
    total_true_pix_size, total_all_pix_size = 0,0
    
    for filename in files:
        #print(filename)
        ind, true_pix_size, all_pix_size = get_train_pixels_coord(folder + '/' + filename, pixels_fraction)

        total_true_pix_size += true_pix_size
        total_all_pix_size += all_pix_size
        
        img_name = filename.split('.npy')[0] + '.png'
        
        img = cv2.imread(train_folder + '/' + img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        new_pixels = img[ind[:,0], ind[:,1]]
        matrix = np.vstack((matrix, new_pixels))
       
    matrix /= 255
    prior = total_true_pix_size / total_all_pix_size
    
    np.save(folder.split('labeled_data_')[1] + '_pixels', matrix)
    np.save(folder.split('labeled_data_')[1] + '_priors', prior)

def create_distance_model():
    files = os.listdir(barrels_folder)
    a = np.zeros((0,1))
    d = np.zeros((0,1))
    for filename in files:
        pixels = np.load(barrels_folder + '/' + filename)
        area = sum(sum(pixels==True))
        dist = filename.split('.npy')[0]
        a = np.vstack((a,area))
        d = np.vstack((d,dist))

    a = np.apply_along_axis(lambda s: float(s), axis=1, arr=a)
    d = np.apply_along_axis(lambda s: float(s), axis=1, arr=d)

    # poor label name for one image: 2_3.1 -> 23.1
    if 23.1 in d:
        d[np.where(d == 23.1)] = 2.31

    a = a.reshape(-1,1)
    d = d.reshape(-1,1)

    # relationship between distance and area is not linear, so we add SQRT feature
    a_train = np.hstack((a,np.sqrt(a)))
    
    lr = LinearRegression().fit(a_train, d)
    joblib.dump(lr, 'distance_model.sav')
    '''
    # Visualize linear regression model fit for distance estimation
    d_hat = lr.predict(a_train)
    plt.scatter(a, d_hat, label='Fitted')
    plt.scatter(a, d, label='Actual')
    plt.ylabel('Distance')
    plt.xlabel('Area in pixels')
    plt.legend()
    plt.show()
    '''
    
    
    

if __name__ == '__main__':
    create_train_file('labeled_data_red',1)
    create_train_file('labeled_data_almost_red',1)
    create_train_file('labeled_data_not_red',0.1)
    create_distance_model()
