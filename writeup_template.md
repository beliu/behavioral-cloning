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
* The 1st layer outputs 16 filters and uses 8x8 filters, with stride 4, 'same' border padding, followed by 'RELU' activation.
* The 2nd layer outputs 32 filters and uses 5x5 filters, with stride 2, 'same' border padding, followed by 'RELU' activation.
* The 3rd layer outputs 64 filters and uses 5x5 filters, with stride 2, 'same' border padding.

* The output of the 3rd convolution layer is flattened and a dropout of 0.2 is applied. A 'RELU' activation follows the dropout.
* The next layer is a fully-connected layer of size 512, followed by a dropout of 0.5, and then a 'RELU' activation.

* The output layer is a fully-connected layer of size 1, which is the prediction of the steering angle. 

####2. Model parameter tuning

I used an adam optimizer for evaluating the model. I output the model loss and accuracy when training so I could see how different parameters affected the model performance. Some of the paramters are tuned were the dropout rate, the batch_size, and the number of epochs. I found that using dropout rates 0.2 and 0.5, a batch size of 64, and, 5 epochs helped me navigate the entire track without running off the road.

####3. Appropriate training data

My training data included data for normal driving and turning around the track for several laps, recovery data that shows the car starting from a position too close to either the right or left edge and then driving back to the center, and a few laps and recovery data from another track. For recovery, I included instances where the car was just nearing the egde and instances where the car was almost about to run off the road. I included several different angles at which the car was approaching the edge. I used images from all three cameras, but I treated the left and right images as if they were from the center camera. I adjusted the steering angle for the left by adding an offset of 5 degrees and the right by subtracting an offset of 5 degrees. 
I originally included rotated, shifted, and brightened or darkened images to augment the dataset but found that it made training the model much longer yet it did not produce much better driving. Therefore I opted to not include augmented data in the training set.

###Model Architecture and Training Strategy

####1. Solution Design Approach

My strategy for developing a model architecture was to design one that can keep the car on the road while reducing overfitting and can run relatively quickly on my laptop.

I adopted the model developed by [commaai](https://github.com/commaai/research/blob/master/train_steering_model.py), since it had a smaller number of layers and was fairly simple. I believed that if the model works, then simplicity should be an advantage. When I tested it first using only the driving data provided by Udacity, the model did not work well at keeping the car on the road. I tried again on the same data with the more complicated [nVidia model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) but it also performed poorly. The car was slow to respond to turns and when it was close to the edge of the raod, it failed to recover. I then proceeded to include more training data and recovery data for the commaai architecture and found that the car was able to drive without going off the road. 

I split my dataset into a training and validation set, with 20% of the dataset going into validation. I set the model to output the MSE loss and accuracy of the model for both the training and validation set after every epoch. The MSE and the accuracy for both the training and validation set were similar, so I believe overfitting was reduced by my approach.

####2. Attempts to reduce overfitting in the model

I believe overfitting would result from having too many images with the same steering angle, not enough 'recovery' data (i.e. data that shows the car recovering to the center of the lane when it drifts too dangerously to the right or left of the road), and only having track one data. 

To reduce the first issue, I randomly sampled only 25% of the total data where the steering angle is 0, which is the most common steering angle output. For the second issue, I recorded data that shows the car starting at some position near the edge of the road and then correcting itself to the center. I included data from both sides of the lane, and included as many different angles of the car to the edge as possible. For the last issue, I included a few thousand data points from track two to the input data.

Within the model itself, I attempted to reduce overfitting by adding in dropout layers after the convolution layers. 

####2. Final Model Architecture

The architecture details are as follows:
The input to the model consists of RGB images of size (90, 320, 3).

The model normalizes the input data to within a range of (-1, 1). Normalization is done using Kera's lambda function.

There are three convolution layers that follow normalization:
* The 1st layer outputs 16 filters and uses 8x8 filters, with stride 4, 'same' border padding, followed by 'RELU' activation.
* The 2nd layer outputs 32 filters and uses 5x5 filters, with stride 2, 'same' border padding, followed by 'RELU' activation.
* The 3rd layer outputs 64 filters and uses 5x5 filters, with stride 2, 'same' border padding.

* The output of the 3rd convolution layer is flattened and a dropout of 0.2 is applied. A 'RELU' activation follows the dropout.
* The next layer is a fully-connected layer of size 512, followed by a dropout of 0.5, and then a 'RELU' activation.

* The output layer is a fully-connected layer of size 1, which is the prediction of the steering angle. 

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
