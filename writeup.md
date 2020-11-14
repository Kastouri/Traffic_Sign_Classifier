# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogramm.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_signs/30.png "Traffic Sign 1"
[image5]: ./test_signs/50.png "Traffic Sign 2"
[image6]: ./test_signs/80.png "Traffic Sign 3"
[image7]: ./test_signs/rutsch.png "Traffic Sign 4"
[image8]: ./test_signs/stop.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

The submitted files include:
* Ipython notebook with code
* HTML output of the code
* A writeup report (markdown)

A Summary and an exploratory visualization of the dataset can be found in section Data Set Summary & Exploration.

A discription of the model architecture and how it was trained can be found in the section Design and Test a Model Architecture.

### Data Set Summary & Exploration


I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a Hisogtram showing the number of images in each class of the training and test dataset.


![alt text][image1]

### Design and Test a Model Architecture

#### 1. Data Preprocessing
I didn't  convet the images to gray scale because the color can be used as a criteria to differentiate some images.

I normalized the image data because a high variance in the data set makes training the model harder. The nomalization was done using the flowwing formula:
	image = (image - 128)/ 128 


#### 2. Model architecture 

My final model consisted of the following layers:

| Layer         		|     Description	        					
|:---------------------:|:---------------------------------------------:
| Input         		| 32x32x3 RGB image   							
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 
| RELU			|											
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16      									
| RELU			|									
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16      									
| RELU			|									
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				
| Convolution 3x3	| 1x1 stride, valid padding, outputs 3x3x64    
| RELU			|									
| Fully connected	| outputs 120		
| Fully connected	| outputs 84
| Fully connected	| outputs 43

#### 3. Training the model

To train the model, I used an Adam optimizer to reduce the mean of the softmax cross entropy.
The following hyperparameter were used:
* number of epochs: 100
* batch size: 128
* learning rate: 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 96.1% 
* test set accuracy of 94.1%

If an iterative approach was chosen:
* Lenet from the lenet lab was chosen
* The initial architecture was underfitting the data
* The learning rate was tuned 
* The number of convolution layer was increased from 2 layers to 3 layers

### Test a Model on New Images

#### 1. Five German traffic signs found on the web

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The  30 km/h image might be difficult to classify because I modified it to simulate a partial obstruction of the sign.

#### 2. Model's predictions on  new traffic signs
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Slippery road     			| Slippery road 										|
| 30 km/h			| 30 km/h										|
| 50 km/h	      		| 50 km/h					 				|
| 80 km/h			| 80 km/h      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.1%

#### 3. Prediction Certainty

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the 80 km/h image, the model is relatively sure that this is a stop sign (probability of 99.9870  % ), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.9870  %         			| 80km/h  									| 
|  0.0130  %     				| 50km/h										|
| 0%					| 100km/h											|
| 0%	      			| 60km/h					 				|
| 0%			    | No vehicles      							|


For the other images the model almost 100% certain. 

