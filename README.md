# ISP 2023 Final Project

## Members

| Number | Student ID |    Name     |
| :----: | :--------: | :---------: |
|   0    | 2021-11566 | Jinyong Jun |
|   1    | 2023-62774 | Claire Kim  |



## Goals

The main objective of this project is to develop a face blurring program that offers real-time face detection, tracking, and automated blurring capabilities. Additionally, the program includes a user interface to allow manual selection and blurring of faces, ensuring privacy compliance when recording video content involving individuals who have not given consent. 



## Technologies

### Face Detection

The face detector included in the dlib library is implemented with HOG(Histogram of Oriented Gradients) and Linear SVM(Support Vector Machine). 

[Face detection with dlib (HOG and CNN) - PyImageSearch](https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/)

### Face Landmark Detection

Face landmark detector is implemented with ensemble of regression tree method. It can detect facial landmarks in real-time with, because they are estimated directly from pixel intensities themselves (i.e., no feature extraction is taking place). 

[Facial landmarks with dlib, OpenCV, and Python - PyImageSearch](https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)

### Face Recognition

Face recognition model is inspired from ResNet-34 model. Regular ResNet structure is modified and some layers are dropped. It represents face images as 128 dimensional vectors. 

[Face Recognition with Dlib in Python - Sefik Ilkin Serengil (sefiks.com)](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/)



## Implementation

### Automated Face Tracking for Main Person

It continuously tracks the designated main person's face throughout the video recording. 



### Selective Face Blurring for Bystanders

It allows users to apply an adjustable blurring effect on the selected faces for privacy protection of passerby. 



### User Interface for Main Person Tracking and Selective Face Blurring

Intuitive user interface allows the user to manually select the main person whose face should be tracked and bystanders whose face should be blurred. 



## Results

TODO



## Discussion

### Challenges

TODO



### Improvements

TODO



### Limitations

TODO



### Future Works

TODO



## Conclusion

TODO



## References

TODO

