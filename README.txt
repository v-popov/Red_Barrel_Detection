---Instructions for Running---

1) Put test images into "Test_Set" folder
2) Simply run the detection.py script and press 0 to go to next image.

---------------------------------------------------------------------------------------------------------------------------------------------------------

--- Files Descriptions---

File labeling.py was used to crop the regions of interest from the training data via roipoly.py library. 
The selected regions were saved to folders labeled_data_[class]. 
Regions in folder barrels were used to calculate the distance function.

File create_train_matrices.py was used to transform selected regions into YCrCb format and for every class all the pixels 
were stacked into a matrix with 3 columns containing 3 color components for each pixel. 
Also here I calculated the priors based on relative region area and calculated the distance function (linear regression model was used). 
All these files were saved as [class]_pixels.npy, [class]_priors.npy and distance_model.sav.

File detection.py is the file that is used to display final results. 
It utilizes the predict.py file where all the logic described in paragraph 2 (Localizing the barrel - will be available in the write-up) is implemented.

---------------------------------------------------------------------------------------------------------------------------------------------------------