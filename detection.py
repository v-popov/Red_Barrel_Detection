import os, cv2
import numpy as np
import predict
from sklearn.externals import joblib

folder = 'Test_Set'

if __name__ == '__main__':
    
    red_matrix = np.load('red_pixels.npy')
    almost_red_matrix = np.load('almost_red_pixels.npy')
    not_red_matrix = np.load('not_red_pixels.npy')
    distance_model = joblib.load('distance_model.sav')

    mu_red, cov_red = predict.estimate_theta(red_matrix)
    mu_almost_red, cov_almost_red = predict.estimate_theta(almost_red_matrix)
    mu_not_red, cov_not_red = predict.estimate_theta(not_red_matrix)
    
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img,x,y,d = predict.predict(img, filename,
                            mu_red, cov_red,
                            mu_almost_red, cov_almost_red,
                            mu_not_red, cov_not_red,
                            distance_model)
        cv2.imshow('image',img)

        print('Test image: ' + filename +
              ', CentroidX = ' + str(x) +
              ', CentroidY = ' + str(y) +
              ', Distance = ' + str(d))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
