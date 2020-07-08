# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals of this project were the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/keepleft.jpg "Traffic Sign 1"
[image4]: ./examples/prio.jpg "Traffic Sign 2"
[image5]: ./examples/prio_road.jpg "Traffic Sign 3"
[image6]: ./examples/warning.jpg "Traffic Sign 4"
[image7]: ./examples/giveway.jpg "Traffic Sign 5"
[image8]: ./examples/noentry.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:
* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed for the labels. We see that there are certain classes having a good number of examples and certain classes having lesser examples for the training.
The training examples contain 43 classes => 43 road signs can be detected.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color information doesnt seem to be necessary for the detection of the signals. Reducing the 3 channel to 1 channel reduces the computation time necessary, preceisely the same reason for which the normalisation was performed on the data.

Here is an example of a traffic sign image before and after grayscaling and normalization.

![alt text][image2]


#### 2. Final model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 14x14x6	|
| Fully connected		| 400->120     									|
| RELU  				|           									|
| Fully connected		| 120->84     									|
| RELU  				|           									|
| Fully connected		| 84->43     									|
| RELU  				|           									|
 


#### 3. Trained model and parameters.

To train the model, I used the same Adam optimizer from the MNIST project. After tuning without additional dataset creation, i ended up with a batch size of 10, number of epochs is 15 though i reached the target of 0.93 almost in the 8th Epoc, and a learning rate of 0.000475.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.946
* test set accuracy of 0.924

The architecture used is the LeNet architecture which is a pretty powerful architecture used for the MNIST example.

I started with using the RGB image and the same parameters as the MNIST and saw an accuracy of 0.83. Then i performed the grayscale conversion and normalisation of the images. This time , the accuracy did not show much improvement and i ended up with an accuracy of around 0.86 maximum. Looking at the variation of the training accuracy and validation accuracy, i could see that the model is still not in overfitting mode. Thus , i decided to experiment with the reduction in learning rate and the batch size. The simultaneous reduction of the learning rate and batch size showed improvement in the accuracy of the validation samples.
The target value of 0.93 accuracy could though not be reached until i added a dropout into the architecture. The addition of a keep probability of 0.5 , improved the accuracy of the model on validation samples to above 0.93.
 

### Testing the Model on New Images

#### 1. German traffic signs found on the web 

Here are six German traffic signs that I found on the web:
The images from the web were skewed on the vertical and horizontal axis before feeding to the model for prediction.

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7] ![alt text][image8]

#### 2. Models predictions

Here are the results of the prediction:

| Image			            |     Prediction	        					| 
|:------------------    ---:|:---------------------------------------------:| 
| Keep left         		| Keep left   									| 
| Right of way     			| Right of way 									|
| Priority road				| Priority road									|
| General caution	   		| General caution								|
| Yield         			| Yield             							|
| No Entry  				| No Entry										|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For all the six images, the models were pretty sure of the images.
Below is the list of top 5 probabilities of the given test images.

Image 1 - 9.99991059e-01,   8.92774278e-06,   2.83060793e-08, 3.32335648e-09,   1.88016025e-09
Image 2 - 9.99999285e-01,   6.63554204e-07,   4.53553532e-15, 2.25422953e-15,   4.32260145e-18
Image 3 - 1.00000000e+00,   8.52744004e-12,   2.23839574e-15, 1.85038667e-15,   2.24527894e-16
Image 4 - 1.00000000e+00,   4.51433488e-18,   1.14463265e-27, 1.05286461e-28,   8.27645072e-31
Image 5 - 1.00000000e+00,   1.38724774e-08,   2.48630533e-10, 3.33660291e-13,   1.66952633e-13
Image 6 - 1.00000000e+00,   2.50106758e-11,   4.65316501e-12, 1.14919916e-14,   5.43666655e-20

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Keep left   									| 
| .99     				| Right of way 									|
| 1.00					| Priority road									|
| 1.00	      			| General caution				 				|
| 1.00				    | Yield             							|
| 1.00				    | No Entry          							|