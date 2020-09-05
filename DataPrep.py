'''
DataPrep
Handles the creation of the ABT
Functions
populates_ABT()- Populates de Analytics Base Table from read images and tags
generates_training_test_sets -From ABT it generates a randomized training and test set
creates_instance(image,tag)-Creates a data instance from an image and a given tag (Tumor or Healthy)
processes_images(dataset)-Reshapes every image in a given dataset to a given size and normalizes pixels
computeKeypointsAndDescriptors()-computes image descriptors and keypoints using ORB (cv2)
loadImageSet()-Loads and returns the image dataset in distinct lists according to the image database
'''
import cv2 as cv
import DataObj
import glob
import os
import random




def populates_ABT():
    healthy_dataset, tumor_dataset = load_image_set()
    processed_healthy_dataset = processes_images(healthy_dataset)
    processed_tumor_dataset = processes_images(tumor_dataset)
    counter = 1
    ABT = DataObj.ABT()
    for image in processed_healthy_dataset:
        tag = 'healthy'
        entry = counter
        keypoints, descriptors = compute_keypoints_and_descriptors(image)
        instance = creates_instance(entry, keypoints, descriptors, image, tag)
        ABT.addinstance(instance)
        counter += 1
    counter = 1
    for image in processed_tumor_dataset:
        tag = 'tumor'
        entry = counter
        keypoints, descriptors = compute_keypoints_and_descriptors(image)
        instance = creates_instance(entry, keypoints, descriptors, image, tag)
        ABT.addinstance(instance)
        counter += 1
    return ABT


def generates_training_test_sets(ABT):
    random.shuffle(ABT)
    training_set = []
    test_set = []
    for instance in ABT:
        dice = random.randint(0, 100)
        if dice >= 30:
            training_set.append(instance)
        else:
            test_set.append(instance)
    return training_set, test_set


def creates_instance(entry, tag, image, keypoints, descriptors):
    instance = DataObj.Instance(entry, tag, image, keypoints, descriptors)
    return instance


def processes_images(images):
    processed_images = []
    width = 600
    height = 600
    for image in images:
        image_height = image.shape[0]
        image_width = image.shape[1]
        if image_height != height or image_width != width:
            processed_image = cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)
            processed_images.append(processed_image)
        else:
            processed_images.append(image)
    return processed_images


def compute_keypoints_and_descriptors(image):
    orb = cv.ORB().create()
    (keypoint, descriptors) = orb.detectAndCompute(image,None)
    return keypoint, descriptors


def load_image_set():
    tumor_brain_dataset = []
    healthy_brain_dataset = []
    img_tumor_dir = r'C:\Users\joseb\Desktop\Documents\Special Projects\Pax Aurora\Analytics\Projects\3.0-Theseus\V0.0\RoboDoc V0\images\yes'
    img_healthy_dir = r'C:\Users\joseb\Desktop\Documents\Special Projects\Pax Aurora\Analytics\Projects\3.0-Theseus\V0.0\RoboDoc V0\images\no'
    tumor_data_path = os.path.join(img_tumor_dir, '*g')
    healthy_data_path = os.path.join(img_healthy_dir, '*g')
    tumor_files = glob.glob(tumor_data_path)
    healthy_files = glob.glob(healthy_data_path)
    for file in tumor_files:
        image = cv.imread(file, 0)
        tumor_brain_dataset.append(image)
    for file in healthy_files:
        image = cv.imread(file, 0)
        healthy_brain_dataset.append(image)
    return healthy_brain_dataset, tumor_brain_dataset


