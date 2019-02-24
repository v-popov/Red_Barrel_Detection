import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal as mvn
from scipy.ndimage.morphology import binary_fill_holes
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from skimage.measure import label, regionprops


# Change of coordinates from XY to RC in skimage package caused a lot of warnings
import warnings
warnings.filterwarnings("ignore")


barrels_folder = 'barrels'
test_folder = 'Test_Set'
#output_data_folder = 'output'


def estimate_theta(X):
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    return mu, cov


def segment_barrel(pixels):
    label_img = label(pixels, connectivity=1)
    max_area = 0;
    for region in regionprops(label_img):
        if region.minor_axis_length > 0:
            sides_ratio = region.major_axis_length / region.minor_axis_length
            if 1.3 < sides_ratio and sides_ratio < 3 and region.area > max_area:
                max_area = region.area
                barrel = region
    return barrel


def localize_barrel(img, filename, output, distance_model):
    # Segmenting the barrel
    barrel = segment_barrel(output)
 
    # Creating barrel mask
    coords = barrel.coords
    x_coord = coords[:,0]
    y_coord = coords[:,1]
    
    image = np.zeros((img.shape[0], img.shape[1]),np.uint8)
    image[x_coord, y_coord] = 255

    # Filling holes inside the barrel
    image = binary_fill_holes(image)
    x_coord, y_coord = np.where(image == 1)

    image = np.zeros((img.shape[0], img.shape[1]),np.uint8)
    image[x_coord, y_coord] = 255

    # Finding barrel's bounding box
    ret,thresh = cv2.threshold(image,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    image = cv2.drawContours(img,[box],0,(0,255,0),2)   

    # Visualizing the barrel's center
    x_cent, y_cent = (box[0] + box[2]) // 2       
    image = cv2.circle(img,(x_cent, y_cent), 3, (0,255,0), -1)
    
    # Computing the distance to barrel
    distance = distance_model.predict(np.array([barrel.area, np.sqrt(barrel.area)]).reshape(1, -1))

    output_text = 'Test image: ' + filename + ', CentroidX = ' + str(x_cent) + ', CentroidY = ' + str(y_cent) + ', Distance = ' + str(round(distance[0][0],1))
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image,output_text,(0,image.shape[0]-20), font, 1, (0,255,0), 2)
    
    # Saving the localized barrel and its center to the output folder
    #cv2.imwrite(output_data_folder + '/' + filename, image)
    
    return image, x_cent, y_cent, (round(distance[0][0],1))
    

def predict(img, filename,
            mu_red, cov_red,
            mu_almost_red, cov_almost_red,
            mu_not_red, cov_not_red,
            distance_model): # -> detect color
    
    # Loading priors
    prior_red = np.load('red_priors.npy')
    prior_almost_red = np.load('almost_red_priors.npy')
    prior_not_red = np.load('not_red_priors.npy')
    
    # Reading an image and converting it to 2D
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) 
    
    img_2d = img.reshape(-1,3) 
    img_2d = np.divide(img_2d,255) # normalization
    X = img_2d

    # Calculating conditional probabilities
    cond_prob_red = mvn.pdf(X, mu_red, cov_red, allow_singular=True)
    cond_prob_almost_red = mvn.pdf(X, mu_almost_red, cov_almost_red, allow_singular=True)
    cond_prob_not_red = mvn.pdf(X, mu_not_red, cov_not_red, allow_singular=True)

    # Calculating posterior probabilities (proportionate values)
    prob_red = cond_prob_red * prior_red
    prob_almost_red = cond_prob_almost_red * prior_almost_red
    prob_not_red = cond_prob_not_red * prior_not_red
    
    # Stacking conditional probabilities in one matrix
    all_probs = np.hstack((prob_red.reshape(-1,1), prob_almost_red.reshape(-1,1)))
    all_probs = np.hstack((all_probs, prob_not_red.reshape(-1,1)))

    # Obtain coordinates of red pixels
    argmax = np.argmax(all_probs, axis=1)
    output = (argmax == 0).reshape(img.shape[0],img.shape[1])

    # Barrel localization
    return localize_barrel(img, filename, output, distance_model)

    
    
