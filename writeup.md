## Writeup

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: examples/training_data.png
[image2]: examples/hog_ex.png
[image3]: examples/window_scale.png
[image4]: examples/sld_windows.png
[image5]: examples/heat_map.png
[image2-1]: examples/2-sld_windows.png
[image2-2]: examples/2-window_scale.png
[image2-3]: examples/2-heatmap.png
[image3-1]: examples/3-only_hog.png
[image3-2]: examples/3-only_hog2.png
[image4-1]: examples/4-sld_windows.png
[image4-2]: examples/4-window_scale.png
[image4-3]: examples/4-heatmap.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

-----

# Updated submission - Oct 21, 2017

My original submission was not good enough and was trigerring too much false positive.
I have tried the following to improve my submission

## [More features and better training](Vehicle+Detection-2.html)

The first thing I have tried is to add more features:
- binned color features,
- color histogram features.

Also I have used the `GridSearchCV` function when training the SVC in order to find the best kernel and C parameter.

As you can see on the following images and videos, the results are vastly improved over the first submission:

![alt text][image2-1]
![alt text][image2-2]
![alt text][image2-3]

Here's the [test video result](./output_images/2-test_video.mp4)

Here's the [project video result](./output_images/2-project_video.mp4)

Overall the result is much better than the first submission but there are still a few false positive.
The improvement comes with some cost. The training is much longer because there are more features and more SVC parameters being optimised. While this is significative, the training cost is paid once and upfront. The processing cost is much more limiting as a frame is processed in about 18s on my PC which is something that should be improved in this submission.

## Trying to speed up by dropping features

My first idea to speed up the training and detection was to drop the binned color and color histogram features. While the training is faster the error rate is too high:

![alt text][image3-1]
![alt text][image3-2]

## [Trying to speed up by optimizing some of the features](Vehicle Detection-Optim.ipynb)

As I described in the previous section, dropping the binned color and the color histogram entirely was giving very bad results with a lot of false positive.

My next idea was to optimize the number of features by:
- changing the size for the binned color: 32x32, 16x16, 8x8, and dropping,
- compute the color histogram for the Y channel only,
- ...

By optimizing the number of features, the training and detection times should both decrease but we want to keep a good detection rate.

Below are the different runs I did to find some good and fast parameters

I started by disabling `GridSearchCV` to get a reference point.
Training time was 291s for a score of 0.990901.

Then I conduct the following trials:

- binned colors 16 * 16: 267s / 0.989764
- hist on Y only: 273s / 0.989195
- cell_per_block = 1: 221s / 0.989764
- spatial off: 94s / 0.951948
- binned colors 8 * 8: 71s / 0.98294

I was quite happy with the latest trial which is much faster to train (71s vs 291s) but still has a good accuracy 98.3% vs 99.1%.

Re-enabling `GridSearchCV` for training gives a final training time of 287s for an accuracy of 99.1%

The result seems quite good on still images:

![alt text][image2-1]
![alt text][image2-2]
![alt text][image2-3]

A frame only take ~3s to be processed which is a 6x improvement over the first implementation.
However the number of false positive is also greatly increased.
Tuning the heatmap parameter to reduce false positive also affects the ability to detect a car.

Here's the [test video result](./output_images/4-test_video.mp4)

Here's the [project video result](./output_images/4-project_video.mp4)


## Conclusion / discussion

The first implementation of this new submission achieves a good quality level but still could be improved. There are some false positive in the middle of the road when there is shade. Having more traning samples would probably help here.

In the absence of more training samples, data augmentation could be used, ie by flipping the training samples left to right and by changing the light condition on the samples.

Hard negative mining as suggested in the previous review would probably help.

Instead of working on the accuracy I decided to work on the speed as the first implementation needs about 18s per frame on my machine and training time is also quite big.

I have tried several different things in the second and third implementation but they all lead to more false positives.

One thing I haven't tried is to decrease the resolution of the input image (ie halving it).

There are many parameters that can be tuned in order to achieve better accuracy or better speed and I've learned in this project that it can be tedious to find the best combination.

While working on this project I have also learned about YOLO and SSD which use a NN approach for object detection. It seems lo lead to very good result and the benefit is that most tunable parameters would be learned during the learning process.

-----

# Original Submission - Oct 1, 2017

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.
Below is an exemple of the training data from both classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the RED channel and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I first tried using the different color spaces and it turned out that the H channel from HLS was giving the best results all other parameter being constant.

Once H(LS) was figured out I tried tuning the others parameters and found the best to be:
- orient 9
- pix_per_cell 10
- cell_per_block 2

In the end I used all the channels of the HLS color space to get better results.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

 I trained a linear SVM with the default classifier parameters and using the HOG alone.
 The test accuracy was good at about 96% but in the end it did generate a lot of false positive.
 It probably would have been a good idea to add more features (spatial intensity & binning).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have used the code from the course because it compute the HOG only once and then sub-sample the result for each window.

I have visually tuned the window scale according to the distance to the car. The closer the car the bigger the scale should be.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

![alt text][image4]

I haven't tried to optimize the perfomance of the classifier.
My goal was to try and optimize the pipeline for correctness but I ran out of time.
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I first deciced to use the HOG only.
I made the choice because I initially thaught a ~96% accuracy would be good enough to give great results.
Turned out that the whole pipeline is very noisy and generate a lot of false positive.

I ran out of time to get a better result.
At least I learned that feature engineering is really important and not so easy.

It would be interesting to compare with an approach that uses deep learning with convolution layers. I guess the feature engineering could be replaced by learning.
