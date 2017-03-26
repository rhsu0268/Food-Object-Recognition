# import the necessary packages
import numpy as np 
import cv2

class BurgerMatcher:
    def __init__(self, descriptor, burgerPaths):
        self.descriptor = descriptor 
        self.burgerPaths = burgerPaths
        
    # keypoints and descriptors extracted from the query image 
    def search(self, queryKps, queryDescs):
        # initialize the dictionary of results
        results = {}
        
        # loop over the burger images
        for burgerPath in self.burgerPaths:
            # load the query image, convert it to grayscale, and extract
            # keypoints and descriptors
            burger = cv2.imread(burgerPath)
            gray = cv2.cvtColor(burger, cv2.COLOR_BGR2GRAY)
            (kps, descs) = self.descriptor.describe(gray)
            
            # determine the number of matched, inlier keypoints, 
            # then update the results
            score = self.match(queryKps, queryDescs, kps, descs)
            results[burgerPath] = score
            
        # if matches were found, sort them 
        if len(results) > 0:
            results = sorted([(v, k) for (k, v) in results.items() if v > 0], reverse = True)
            
        # return the results
        return results
            