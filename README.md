# RoboDoc-V 0.0
===============

Introduction

RoboDoc is a image matching program that uses KNN to find abnormalities in brain scans. It can tell whether a scan shows a tumor or a healthy scan. 

Setup
--------------
1) Download all python modules and the folder 'images' containing brain scans for training 
2) In the DataPrep.py module locate the load_image_set() function and update the path to the 'images' folder in your drive

Usage
---------------
1) Run main() a GUI should pop up
2) Use 'Load Image' button to load an image from your local drive
3) Use 'Train model' button to train the knn algorithm
4) Use 'Match Query' to find the best match for the query image from the training DataSet.


