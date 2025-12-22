                                  Real Time Face with Age-Gender Detection using jetson AI
                                  --------------------------------------------------------

# Problem Statement:
--------------------

Airports handle large volumes of passengers, making manual security monitoring inefficient and error-prone. Existing systems often lack real-time automated demographic analysis, and cloud-based solutions introduce latency, bandwidth costs, and privacy concerns.
There is a need for a fast, reliable, edge-based AI system that can:
Automatically detect passengers
Analyze age and gender in real time
Operate with low latency
Preserve data privacy by avoiding cloud processing
This project addresses these challenges by implementing a Jetson Nano–based real-time age and gender detection system for airport security applications

# Project Objectives:
---------------------
Design a real-time system to detect passenger age and gender
Implement an edge-based AI solution using Jetson Nano
Reduce latency and privacy risks by avoiding cloud processing
Support airport security and crowd monitoring use cases
Demonstrate a scalable and practical computer vision pipeline

# System Architecture:
----------------------
The system follows the pipeline below:
Input: USB camera captures live video frames
Face Detection: Haar Cascade detects faces in each frame
Centroid Tracking: Assigns a unique ID to each detected face
People Counting: Counts unique tracked IDs
Preprocessing: Converts face ROI to blob (227 × 227)
Inference Models:
GenderNet → Male / Female
AgeNet → Age range bucket
Output: Displays bounding box, ID, age, gender, and people count on live video

# Workflow:
-----------

Capture live video frames using a USB camera
Detect faces using Haar Cascade classifier
Apply centroid tracking to track faces across frames
Assign a unique ID to each detected person
Extract face ROI for each tracked person
Preprocess face images to 227 × 227 input size
Perform age and gender prediction using deep learning models
Display bounding boxes, ID, age, gender, and people count on live video

# Models Use:
-------------
--AgeNet
Purpose: Predict age range
Output: Predefined age group buckets
Accuracy: 65–70%
Framework: Caffe
Execution: OpenCV DNN module
--GenderNet
Purpose: Gender classification
Output: Male / Female
Accuracy: ~85%
Framework: Caffe
Execution: OpenCV DNN module
Both models are pre-trained Caffe-based models, optimized for real-time inference on Jetson Nano 

Centroid Tracking Algorithm
---------------------------
Centroid tracking is a lightweight object tracking algorithm used to track detected faces across consecutive video frames.
The centroid (center point) of each face bounding box is calculated
Centroids from the current frame are compared with previous centroids
If a matching centroid is found within a distance threshold, the same ID is retained
If no match is found, a new ID is assigned
Prevents counting the same person multiple times

Why Centroid Tracking?
----------------------
Computationally efficient
Suitable for real-time edge devices
Simple to implement
Effective for moderate movement scenarios
This algorithm enables reliable people counting and ensures demographic predictions remain associated with the correct individual

Technologies Used:
------------------
-Python
-OpenCV
-NumPy
-Haar Cascade Classifier
-Deep Learning (Caffe – AgeNet & GenderNet)
-OpenCV DNN module
-NVIDIA Jetson Nano

Project Structure
-----------------

project/
│── project4_update.py
│── README.md
│── models/
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   ├── gender_net.caffemodel
│── demo/
│   └── demo_video.mp4

How to Run the Project
----------------------
python project4_update.py

Results
-------
-Real-time face detection and tracking
-Accurate gender and age range prediction
-Reliable people counting using centroid tracking
-Low-latency edge-based performance
-Clear visualization of demographic information

Video Demonstrates:
-------------------
-Live video capture
-Face detection
-Centroid-based tracking with IDs
-Age and gender prediction
-Real-time output visualization

Video Link:
-----------
[ https://github.com/user-attachments/assets/fc2c43cb-90cf-411a-ae2a-e469875df1fc ]

Limitations:
------------
-Accuracy depends on lighting and face visibility
-Occlusions may affect tracking
-Age prediction is approximate
-Designed primarily for controlled environments

Future Enhancements:
--------------------
-Replace Haar Cascade with CNN-based face detection
-Integrate advanced tracking (DeepSORT)
-Store demographic data in a database
-Deploy as a real-time airport monitoring dashboard
-Improve performance in crowded scenes

Authors
-------
Tejaswini Veeramachaneni ,
Sruthi Velpula

License:
This project is developed for academic purposes only.
