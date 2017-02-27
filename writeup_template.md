#**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network that is a modification of the commaai network architecture. The original
architecture can be found here: [commaai](https://github.com/commaai/research/blob/master/train_steering_model.py)

The architecture details are as follows:
The input to the model consists of RGB images of size (90, 320, 3).

The model normalizes the input data to within a range of (-1, 1). Normalization is done using Kera's lambda function.

There are three convolution layers that follow normalization:
The 1st layer outputs 16 filters and uses 8x8 filters, with stride 4, 'same' border padding, followed by 'RELU' activation.
The 2nd layer outputs 32 filters and uses 5x5 filters, with stride 2, 'same' border padding, followed by 'RELU' activation.
The 3rd layer outputs 64 filters and uses 5x5 filters, with stride 2, 'same' border padding.

The output of the 3rd convolution layer is flattened and a dropout of 0.2 is applied. A 'RELU' activation follows the dropout.
The next layer is a fully-connected layer of size 512, followed by a dropout of 0.5, and then a 'RELU' activation.

The output layer is a fully-connected layer of size 1, which is the prediction of the steering angle. 


####2. Attempts to reduce overfitting in the model

I believe overfitting would result from having too many images with the same steering angle, not enough 'recovery' data (i.e. data that shows the car recovering to the center of the lane when it drifts too dangerously to the right or left of the road), and only having track one data. 

To reduce the first issue, I randomly sampled only 25% of the total data where the steering angle is 0, which is the most common steering angle output. For the second issue, I recorded data that shows the car starting at some position near the edge of the road and then correcting itself to the center. I included data from both sides of the lane, and included as many different angles of the car to the edge as possible. For the last issue, I included a few thousand data points from track two to the input data.

Within the model itself, I attempted to reduce overfitting by adding in dropout layers after the convolution layers. 

####3. Model parameter tuning

I used an adam optimizer for evaluating the model. I output the model loss and accuracy when training so I could see how different parameters affected the model performance. Some of the paramters are tuned were the dropout rate, the batch_size, and the number of epochs. I found that using dropout rates 0.2 and 0.5, a batch size of 64, and, 5 epochs helped me navigate the entire track without running off the road.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
