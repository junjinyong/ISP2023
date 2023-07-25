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

Refer to  [Histograms of Oriented Gradients for Human Detection](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf).

### Face Landmark Detection

The face landmark detector implements an ensemble of regression tree methods. A regression tree is a decision tree used in machine learning that recursively partitions the input space and assigns continuous numerical values to each leaf node, making it suitable for predicting continuous target variables. The ensemble of regression trees in the proposed method combines the predictions from multiple trees using averaging or voting mechanisms to arrive at a final and more accurate landmark position estimation. The method achieves state-of-the-art face alignment within just one millisecond, demonstrating robustness to challenging conditions like varying poses and occlusions because they are estimated directly from pixel intensities without any feature extraction taking place. 

Refer to  [One Millisecond Face Alignment with an Ensemble of Regression Trees](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf) for more details. 

### Face Recognition

The face recognition model takes inspiration from the ResNet-34 model. The regular ResNet structure is modified by dropping several layers and rebuilt to have 29 convolutional layers. It represents face images as 128-dimensional vectors. It expects 150x150x3-sized inputs and represents face images as 128-dimensional vectors. The model is then re-trained for various data sets, including FaceScrub and VGGFace2. In other words, it learns how to find face representations with 3M samples. Subsequently, the model achieved impressive accuracy of 99.38% on the LFW dataset, a widely accepted benchmark for face recognition research. On the other hand, human beings hardly have a 97.53% score on the same dataset. It implies that the dlib face recognition model can compete with other state-of-the-art face recognition models and human beings. 

Refer to  [Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) for more details. 



## Implementation

### Face Recognition Based on Face Embeddings

The dlib library facilitates calculating face embeddings in the following process. First, the embedded frontal face detector detects faces. Then, the shape predictor model obtains 68 facial landmarks for each face. The image is cropped on each face to reduce the computational cost. Last, the face recognition model based on ResNet computes 128-dimensional facial embeddings. 

```python
# Calculate face encoding
shapes = [recognizer.predict(image, location) for location in locations]
landmarks = [recognizer.rescale(image, landmark) for landmark in shapes]
encodings = [np.array(recognizer.encode(landmark, num_jitters=1)) for landmark in landmarks]
```



```python
predictor_location = r"shape_predictor_68_face_landmarks.dat"
recognizer_location = r"dlib_face_recognition_resnet_model_v1.dat"

class FaceRecognizer:
    def __init__(self):
        self.__detector = dlib.get_frontal_face_detector()
        self.__predictor = dlib.shape_predictor(predictor_location)
        self.__rescaler = dlib.get_face_chip
        self.__encoder = dlib.face_recognition_model_v1(recognizer_location).compute_face_descriptor
    
    # (ellipsis)
```



### Face Database for Unknown Face Classification

A new data structure named face database saves face embeddings of known faces. To distinguish faces, it assigns IDs to face embeddings. It can handle queries by comparing queried embedding with all the saved embeddings. If there is someone in the database whose cosine similarity with the queried face is less than the threshold, they are considered the same person. If not, the queried one is a new person and saved into the database. 

```python
class FaceDatabase:
    def __init__(self, threshold=0.4, decay=0.1):
        self.__data = list()
        self.__blur = list()
        self.__threshold = threshold
        self.__decay = decay

    def query(self, face, update=True, insert=True):
        # Initialize minimum and index
        minimum: float = float("inf")
        index: int = -1
        blur: bool = False

        # Calculate minimum and index
        if len(self.__data) > 0:
            distances = np.linalg.norm(self.__data - face, axis=1)
            minimum = min(distances)
            index = int(np.argmin(distances))

        # (ellipsis)

        # Insert new face to the database if the face does not exist
        if insert and minimum > self.__threshold:
            index = len(self.__data)
            self.__data.append(face)
            self.__blur.append(False)

        return index, blur
    
	# (ellipsis)
```



### Automated Face Tracking for Main Person

It continuously tracks the designated main person's face throughout the video recording. Clicking on the desired individual can reassign the main person. The point is to match faces on the screen to the main person's face embedding to find out which face on the screen corresponds to the main person. If the central person's face is in the view, we can track it to be in the center of the image. The main sticking point was that Gretchen made excessive movements. Because the image processing is slow, the device erroneously perceives the movement as insufficient and thus goes a little more. Gretchen proceeds only one-third of desired one with reduced velocity to reduce such phenomena. 

```python
# Calculate the distance to all faces in the database
# 
for (order, face) in enumerate(encodings):
    if database.compare(face, host):
        owner = order
    
    # (ellipsis)

# (ellipsis)

# Track the main person
if owner >= 0:
    look(robot, camera, locations[owner])
```

```python
def look(robot, camera, face):
    # Get 2d coordinates
    u = (face.left() + face.right()) / 2
    v = (face.top() + face.bottom()) / 2

    # Rescale coordinates
    # Countervail the excessive movement of the camera due to the slow feedback
    u = (320 + 320 + u) / 3
    v = (240 + 240 + v) / 3

    # Convert the 2d coordinates to 3d coordinates in camera frame
    (x, y, z) = camera.convert2d_3d(u, v)

    # Convert the 3d coordinates from the camera frame into
    # Gretchen's frame using a transformation matrix
    (x, y, z) = camera.convert3d_3d(x, y, z)

    # have Gretchen look at that point
    robot.lookatpoint(x, y, z, velocity=0.54)
```



### Selective Face Blurring for Bystanders

It allows users to apply an adjustable blurring effect on the selected faces for the privacy protection of passersby. The database stores whether to blur or not each face ID. All the detected faces are queried to the database to determine whether to mask. A smiley face image covers faces to obscure. 

```python
# Query faces
protection: list = list()
for (order, face) in enumerate(encodings):
    # (ellipsis)
    index, blur = database.query(face, update=True, insert=True)
    protection.append(blur)
    
# (ellipsis)

# Blur faces
for (face, blur) in zip(locations, protection):
    if blur:
        image = blurer.blur(image, face)
```



```python
def blur(self, img, face):
    # Make into PIL image
    result = Image.fromarray(img)

    # Get a drawing context
    (size, pos) = dispose(face)
    icon = self.__raw.resize((size, size))

    # Draw emoji on face
    result.paste(icon, pos, mask=icon)

    # Convert back to OpenCV image
    result = np.array(result)

    return result
```



### User Interface for Main Person Tracking and Selective Face Blurring

The intuitive user interface allows users to manually select the central person to track and bystanders whose faces should be blurred. Clicking on the screen calls a callback function and asserts a signal flag. Then, the main loop processes the requests synchronously. It looks for all the faces on the screen and finds the nearest. Nothing happens when there is no face whose distance to the clicked point is less than the tolerance. Then it queries the closest one to the clicked coordinate in the database to get the ID. If the event is left button down, we toggle whether to blur the face. For the right button down, we change the central person to track. 

```python
# Process click events
if signal:
    # Find face nearest to the clicked point
    target = findNearest(point, locations)

    # Query the found face to the database to obtain DB index
    face = encodings[target]
    index = database.query(face, update=False, insert=False)[0]

    # invert whether to blur or not if left button down
    # change main person if right button down
    if signal == 1:
        database.toggle(index)
    elifsignal == 2:
        host = index

signal = 0
```

```python
def findNearest(p, locations):
    # Initialize minimum and index
    minimum: float = 100.0
    index: int = -1
    u, v = p

    # Find the closest face whose distance is less than the tolerance
    for (order, face) in enumerate(locations):
        # Calculate center
        x = (face.left() + face.right()) / 2
        y = (face.top() + face.bottom()) / 2

        # Calculate distance between clicked point and the center of the face
        distance = abs(x - u) + abs(y - v)

        # Update minimum
        if distance < minimum:
            minimum = distance
            index = order

    return index
```



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

Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection.



