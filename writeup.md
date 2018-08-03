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

[image1]: ./images/dataset.png "Image Visualization"
[image2]: ./images/histogram.png "Histogram"
[image3]: ./images/grayscale.png "Grayscale"
[image4]: ./images/normalizedgrayscale.png "grayscale and normalized image"
[image5]: ./images/newimages.png "new traffic signs"
[image6]: ./images/newimage2.png "grayscale and normalized new image"
[image7]: ./images/softmax1.png "softmax of image 1"
[image8]: ./images/softmax2.png "softmax of image 2"
[image9]: ./images/softmax3.png "softmax of image 3"
[image10]: ./images/softmax4.png "softmax of image 4"
[image11]: ./images/softmax5.png "softmax of image 5"
[image12]: ./images/softmax6.png "softmax of image 6"
[image13]: ./images/lenet5.png "lenet5 architecture"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/vishnuGovindaraj/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy methods to obtain the following results

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = 1024
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It shows each image class and the corresponding image count 

![alt text][image1]

Also here is a histogram showing the ClassID and corresponding image count

![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


The two preprocessing techniques I used were grayscale transformation and normalization. After researching online It's apparent that to get a really high accuracy on the validation set, it's necessary to do some type of data augmentation, especially on the MNIST data set where there are a lot of classes that are not as well represented as other classes. I decided to not use any data augmentation, since I didn't have much time, and I think I can comfortably get higher than 93% accuracy with other techniques.

Gray scale transformation is important because it allows several positive effects.

- It's quicker for the computer to process, because we are operating on less information 3 channels to 1.
- It generally makes identifying edges easier, which could be difficult with a color image
- If I want to add data augmentation techniques later on, It's simpler on gray scale images due to less code and faster process time.

Normalization is important because it ensures that the input parameter (pixel) for the network has a similar data distribution, which means that convergence would be faster when training the network. 

The following is a image after being transformed into grayscale

![alt text][image3]

The grayscale image is then normalized

![alt text][image4]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The architecture I chose to implement was the LeNet5 architecture described in class with a few modifications.

![alt text][image13]

The LeNet5 architecture has been modified to receive grayscale images and uses dropout.
 - The original LeNet was set to receive color images, this is modified to 32x32x1 because of grayscale
 - There are 2 convolutional layers, 2 max pooling layers, and 3 fully connected layers. The last fully connected layer is the output. This architecture uses 2 dropout rates after 4 layers. 2 of which are after the convolutional layer, and 2 after the first 2 fully connected layers
 - The first layer is a convolutional layer: Input = 32x32x1. Output = 28x28x6. 
   filter width is 5x5, input depth is 1, output depth is 6. The strides are 1 across, 1 down with valid padding.
 - There is a dropout here of 0.8, so 80% of the activations are retained during training. 100% during validation and testing.
 - The second layer is a max-pooling layer: Input = 28x28x6. Output = 14x14x6.
   filter width is 2x2, The strides are 2 across, 2 down. Padding is valid.
 - The third layer is a convolutional layer: Input = 14x14x6. Output = 10x10x16.
   filter width is 5x5, input depth is 6, output depth is 16. The strides are 1 across, 1 down with valid padding.
 - There is a dropout here of 0.8, so 80% of the activations are retained during training. 100% during validation and testing.
 - The fourth layer is a max-pooling layer: Input = 10x10x16. Output = 5x5x16
   filter width is 2x2, The strides are 2 across, 2 down. Padding is valid.
 - Then the output is flattened so 5x5x16 becomes 400.
 - The 5th layer is a fully connected layer that takes 400 input and produces 120 ouput.
   There is a dropout here of 0.6, so 60% of the activations are retained during training. 100% during validation and testing.
 - The 6th layer is a fully connected layer that takes 120 input and produces 84 output.
   There is a dropout here of 0.6, so 60% of the activations are retained during training. 100% during validation and testing.
 - The final layer is a fully connected layer that takes 84 inputs and produces 43 outputs, one for each type of traffic sign.
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyper parameters such as learning rate.

I experimented with different values for dropout, epochs, learning rate and batch size. I did not modify sigma, mu, filter size and strides or layer size.
I used the optimizer provided in the LeNet lab solution (AdamOptimizer) since it is supposed to be more sophisticated than Stochastic Gradient Descent.

The following is a list of the values I used. Question 4 provides more detail as to how I picked these values.

- Epochs: 100
- LearningRate: 0.001
- BatchSize: 256
- KeepProb1: 0.9
- KeepProb2: 0.5


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My solution approach was to try and find a good value for epochs, batchsize, learningrate and dropout that will give me a validation accuracy of over 93%.

The following are tables where specific qualities are tested

The conclusion after each table is listed:

|Epochs|LearningRate|BatchSize|KeepProb1|KeepProb2|Validation Accuracy
|:--:|:--:|:--:|:--:|:--:|:--:|
|10| 0.001| 128 |1.0|1.0|92%
| 50| 0.001  | 128  |  1.0 | 1.0  |92%





Conclusion: Increasing the number of epochs from 10 to 50 did not increase the accuracy with no dropout

|Epochs|LearningRate|BatchSize|KeepProb1|KeepProb2|Validation Accuracy
|:--:|:--:|:--:|:--:|:--:|:--:|
| 10| 0.0001  | 128  |  1.0 | 1.0  |76%
| 50| 0.0001  | 128  |  1.0 | 1.0  |85%
| 100| 0.0001  | 128  |  1.0 | 1.0  |87%


Conclusion: Learning rate is decreased from 0.001 to 0.0001. Even after 100 epochs the accuracy is not as good as when the learning rate was 0.001


|Epochs|LearningRate|BatchSize|KeepProb1|KeepProb2|Validation Accuracy
|:--:|:--:|:--:|:--:|:--:|:--:|
| 100| 0.001  | 256  |  1.0 | 1.0  |93%
| 100| 0.0001  | 256  |  1.0 | 1.0  |85%


Conclusion: Batch size was increased from 128 to 256. 2 different learning rates are tested and 0.001 gives the better result

|Epochs|LearningRate|BatchSize|KeepProb1|KeepProb2|Validation Accuracy
|:--:|:--:|:--:|:--:|:--:|:--:|
|10| 0.001| 256 |0.8|1.0|86%
|10| 0.001  | 256  |  0.8 | 0.5  |90%


Conclusion: Row 1 has dropout at the convolution layers but not at the fully connected layers. Row 2 has dropout at both the convolutional layers and the fully connected layers. Row 2 has better accuracy


|Epochs|LearningRate|BatchSize|KeepProb1|KeepProb2|Validation Accuracy
|:--:|:--:|:--:|:--:|:--:|:--:|
|50| 0.001| 256 |0.8|1.0|92%
|50| 0.001  | 256  |  0.8 | 0.5  |96%


Conclusion: Epochs is increased from 10 to 50. Row 2 still has better accuracy and is now above the 93% required validation accuracy.

Some more results from varying dropout rates.

|Epochs|LearningRate|BatchSize|KeepProb1|KeepProb2|Validation Accuracy
|:--:|:--:|:--:|:--:|:--:|:--:|
|50| 0.001| 256 |0.9|0.5|96%
|50| 0.001  | 256  |  0.8 | 0.6  |95%
|100| 0.001  | 256  |  0.8 | 0.5  |96%
|100| 0.001  | 256  |  0.8 | 0.6  |96%
|100| 0.001  | 256  |  0.9 | 0.5  |97%
|100| 0.001  | 256  |  0.9 | 0.6  |96%

Conclusion: Varying the dropout rates netted very similar results. The final values I decided to use are as follows:

- Epochs: 100
- LearningRate: 0.001
- BatchSize: 256
- KeepProb1: 0.9
- KeepProb2: 0.5

Note: Running the model with 0.9 and 0.5 did not always give me 97%. But I got consistently 96% with these values.
Also my captured images got 83% accuracy as opposed to 67% with other keepprob values.
 

### Test a Model on New Images

#### 1. Choose six German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I picked 6 images which correspond to the following ClassIds:

|Classid|SignName|
|--:|--:|
|1| Speed limit (30km/h)
|11| Right-of-way at the next intersection
|25| Road work
|26| Traffic signals
|3| Speed limit (60km/h)
|34| Turn left ahead

I picked 2 speed limits to try and see how well the model can differentiate between the 2 similar signs.

I picked 3 signs with a triangular shape to further assess the models effectiveness in determining edges and what is inside the shape. I want to see how confident the model is in predicting which triangular sign is which.

The last image I have is a circular shape (turn left). I want to see the probability of this being a speed limit sign.

I think that the model will be able to distinguish between the speed signs because of how big/distinct the numbers are inside the shape relative to the shape of the sign. 
I think it will have some problems with the triangular images because of the similarities between the images inside the sign.

Note: the images get read in different order when testing on an linux machine (AWS instance), which requires the order of the labels to be changed.

![alt text][image5]

After being transformed the first image looks like this

![alt text][image6]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The accuracy for the newly captured images is 83%, which is lower than the 94% accuracy in the test set. This could be because the model was not trained well for these specific images. Even though it correctly predicted all 6 images, It was not extremely confident about it's predictions which is outlined in the next section


Here are the results of the prediction:

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]


The softmax probabilities shows that the model was 100% sure on all the images except 1. The accuracy result of 83% should be higher considering that all the images were correctly classified.


