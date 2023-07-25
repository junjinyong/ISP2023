# ISP 2023 Final Project

## Members

| Number | Student ID |    Name     |
|:------:|:----------:|:-----------:|
|   0    | 2021-11566 | Jinyong Jun |
|   1    | 2023-62774 | Claire Kim  |



## Abstract

TODO



## Introduction

Due to the increasing reliance on and advancement of technology, there is no doubt that the usage of technology has significantly increased. It plugs into every aspect of life. Sometimes individuals are recorded for security reasons, such as at the airport for identification or CCTV. Other times individuals are recorded, with or without consent, for entertainment -- one prime example being YouTube videos. This project looks at one possibility for approaching this issue. 



## Goals

The main objective of this project is to develop a face-blurring program that offers real-time face detection, tracking, and automated blurring capabilities. Additionally, the program includes a user interface to allow manual selection and blurring of faces, ensuring privacy compliance when recording video content involving individuals who did not consent. 



## Technologies

### Face Detection

The face detector included in the dlib library is implemented with HOG(Histogram of Oriented Gradients) and Linear SVM(Support Vector Machine). It works as follows: 

The first step is to compute HOG features from the input image. HOG works by dividing the image into small cells and computing gradient orientations and magnitudes within each cell. These local gradients create histograms of gradient orientations. The histogram captures the distribution of edge orientations in the cell. HOG descriptors give a compact representation of an image's local texture and gradient information. After computing the HOG features, a sliding window technique is employed to move a fixed-sized window (similar to the size of a face) across the entire image. HOG descriptors are extracted for each window location and fed into a pre-trained linear SVM classifier. The SVM trains to distinguish between face and non-face regions using HOG features. It has learned to distinguish between the typical HOG feature patterns of faces and those that are not. After classifying each window as a face or non-face, the post-processing step, known as non-maximum suppression, is performed to eliminate duplicate detections and retain only the most confident face detections. It regards removing overlapping bounding boxes by selecting the one with the highest SVM confidence score. 

[Face detection with dlib (HOG and CNN) - PyImageSearch](https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/)

### Face Landmark Detection

The face landmark detector implements an ensemble of regression tree methods. A regression tree is a decision tree used in machine learning that recursively partitions the input space and assigns continuous numerical values to each leaf node, making it suitable for predicting continuous target variables. The ensemble of regression trees in the proposed method combines the predictions from multiple trees using averaging or voting mechanisms to arrive at a final and more accurate landmark position estimation. The method achieves state-of-the-art face alignment within just one millisecond, demonstrating robustness to challenging conditions like varying poses and occlusions because they are estimated directly from pixel intensities without any feature extraction taking place. 

One Millisecond Face Alignment with an Ensemble of Regression Trees by Vahid Kazemi and Josephine Sullivan

[Facial landmarks with dlib, OpenCV, and Python - PyImageSearch](https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)

### Face Recognition

The face recognition model takes inspiration from the ResNet-34 model. The regular ResNet structure is modified by dropping several layers and rebuilt to have 29 convolutional layers. It represents face images as 128-dimensional vectors. It expects 150x150x3-sized inputs and represents face images as 128-dimensional vectors. The model is then re-trained for various data sets, including FaceScrub and VGGFace2. In other words, it learns how to find face representations with 3M samples. Subsequently, the model achieved impressive accuracy of 99.38% on the LFW dataset, a widely accepted benchmark for face recognition research. On the other hand, human beings hardly have a 97.53% score on the same dataset. It implies that the dlib face recognition model can compete with other state-of-the-art face recognition models and human beings. 

Deep Residual Learning for Image Recognition

[Face Recognition with Dlib in Python - Sefik Ilkin Serengil (sefiks.com)](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/)



## Implementation

### Automated Face Tracking for Main Person

It continuously tracks the designated main person's face throughout the video recording. 



### Selective Face Blurring for Bystanders

It allows users to apply an adjustable blurring effect on the selected faces for privacy protection of passerby. 



### User Interface for Main Person Tracking and Selective Face Blurring

Intuitive user interface allows the user to manually select the main person whose face should be tracked and bystanders whose face should be blurred. 



## Results

This project is able to identify major facial features (such as the eyes, nose and mouth), and has the ability to utilize an identification system to track different faces concurrently. It also gives the user the option to cover a person's face with an emoji if they desire by simply clicking on the face. 



## Discussion

It was intriguing to witness how the program was able to identify a person's face then later have the ability to recognize and track it. 

### Challenges

TODO



### Improvements

Currently, the face recognition model implemented in this project is only able to recognize faces that are directly facing the camera. A signicifcant improvement to this project would be having the ability to detect side profiles. Additionally, there is only one option to cover a person's face. It is with a smiling emoji. This may be unfortunate to those who may wish to depict other emotions. 



### Limitations

The camera utilized in this project is not able to capture real-time movement accurately. It takes a bit of time for the camera to process what it captures. The camera is also fixated at a specific height, so external adjustments had to be made to correctly capture people's faces. 



### Future Works

TODO



## Conclusion

TODO



## References

TODO

