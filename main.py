# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import interface
'''
ABT = DataPrep.populates_ABT()
precision = 0.5
while precision < 0.83:
    training_set, test_set = DataPrep.generates_training_test_sets(ABT.ABT)
    prediction_set = KNN.test_model(test_set, training_set)
    precision, recall = KNN.compute_metrics(prediction_set)
print('Model Precision: ', precision, 'Model Recall: ', recall)
query_1 = cv.imread('normal-ct.jpg', 0)
query_2 = cv.imread('metastatic-malignant-melanoma.jpg', 0)
query_3 = cv.imread('glioblastoma-base.jpg', 0)

query_instance_1 = KNN.creates_query_instance(query_1, 'healthy')
query_instance_2 = KNN.creates_query_instance(query_2, 'tumor')
query_instance_3 = KNN.creates_query_instance(query_3, 'tumor')

match_1 = KNN.finds_best_match(query_instance_1, training_set)
match_2 = KNN.finds_best_match(query_instance_2, training_set)
match_3 = KNN.finds_best_match(query_instance_3, training_set)

print('Prediction: ' + match_1.prediction, 'Category: ' + match_1.tag, 'File: normal-ct')
print('Prediction: ' + match_2.prediction, 'Category: ' + match_2.tag, 'File: metastatic malignant melanoma')
print('Prediction: ' + match_3.prediction, 'Category: ' + match_2.tag, 'File: glioblastoma')
img1 = cv.drawMatches(match_1.image, match_1.keypoints, match_1.best_match.image, match_1.best_match.keypoints, match_1.k_matches[:10], None, flags=2)
img2 = cv.drawMatches(match_2.image, match_2.keypoints, match_2.best_match.image, match_2.best_match.keypoints, match_2.k_matches[:10], None, flags=2)
img3 = cv.drawMatches(match_3.image, match_3.keypoints, match_3.best_match.image, match_3.best_match.keypoints, match_3.k_matches[:10], None, flags=2)

cv.imwrite('match_1.jpg', img1)
cv.imwrite('match_2.jpg', img2)
cv.imwrite('match_3.jpg', img3)
'''


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    root = interface.Root()
    root.mainloop()
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
