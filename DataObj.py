'''
DataPrep
Data Reading and Preparation
'''


#Data Instance Class
class Instance():
    def __init__(self, entry=0, keypoints=None, descriptors=None, image=None, tag='', prediction='', best_match=None,
                 k_matches=None):
        self.entry = entry
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.image = image
        self.tag = tag
        self.prediction = prediction
        self.k_matches = k_matches
        self.best_match = best_match
        self.instance_row = [self.entry, self.keypoints, self.descriptors, self.image, self.tag]

    def returninstancerow (self):

        return self.instance_row

#Analytics Base Table Class
class ABT():
    def __init__(self):
        self.ABT = []

    def addinstance(self,instance):
        self.ABT.append(instance)

    def getABT(self):
        return self.ABT




