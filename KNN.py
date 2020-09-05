'''
KNN-Module
Executes the KNN search algorithm using the open CV orb brute force matcher
Functions
creates_query_instance(query_image,query_tag)-Creates an Instance Object from a queried image
match_query(k,query,trainingSet)-Returns the k closest matches to a given query instance from the training set
finds_best_match(query_instance,training_set)-finds the best match for a given query from the training set
test_model(testSet,trainingSet)-Evaluates the model by performing brute match for each instance of the training set and
returning a predicted testSet
compute_metrics(testSet)-From the predicted testset compute metrics for model evaluation
'''

import cv2 as cv
import DataPrep


def creates_query_instance(query_image, query_tag):
    query_width = query_image.shape[0]
    query_height = query_image.shape[1]
    if query_width != 600 or query_height != 600:
        query_image = cv.resize(query_image, (600, 600), interpolation=cv.INTER_LINEAR)
    query_keypoints, query_descriptors = DataPrep.compute_keypoints_and_descriptors(query_image)
    query_instance = DataPrep.creates_instance(0, query_keypoints, query_descriptors, query_image, query_tag)
    return query_instance


def match_query(query_instance, training_instance, k=35):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(query_instance.descriptors, training_instance.descriptors)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[:k+1]


def finds_best_match(query_instance, training_set):
    best_match = None
    initial_match = 1000000000
    for instance in training_set:
        k_matches = match_query(query_instance, instance)
        k_square_sum = sum(map(lambda x: x.distance**2, k_matches))
        if k_square_sum < initial_match:
            initial_match = k_square_sum
            best_match = instance
            query_instance.k_matches = k_matches
            query_instance.best_match = best_match
            query_instance.prediction = instance.tag
    return query_instance


def test_model(test_set, training_set):
    prediction_set = []
    for instance in test_set:
        prediction_instance = finds_best_match(instance, training_set)
        prediction_set.append(prediction_instance)
    return prediction_set


def compute_metrics(prediction_set):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for instance in prediction_set:
        if instance.tag == 'healthy' and instance.prediction == 'healthy':
            true_negatives += 1
        elif instance.tag == 'tumor' and instance.prediction == 'tumor':
            true_positives += 1
        elif instance.tag == 'healthy' and instance.prediction == 'tumor':
            false_positives += 1
        elif instance.tag == 'tumor' and instance.prediction == 'healthy':
            false_negatives += 1
    precision = true_positives/(true_positives+false_positives)
    recall = true_positives/(true_positives+false_negatives)
    return precision, recall
