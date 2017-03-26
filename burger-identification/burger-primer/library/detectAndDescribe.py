# import the necessary packages
import numpy as np 

class DetectAndDescribe:
    def __init__(self, detector, descriptor):
        # store the keypoint detector and local invariant descriptor 
        self.detector = detector 
        self.descriptor = descriptor 
        
    def describe(self, image, useKplist=True):
        # detect keypoints in the image and extract local invariant descriptors 
        kps = self.detector.detect(image)
        (kps, descs) = self.descriptor.compute(image, kps)
        
        # if there are no keypoints or descriptors, return None
        if len(kps) == 0:
            return (None, None)
        
        # check to see if the keypoints should be converted to a NumPy array
        if useKplist:
            kps = np.int0([kp.pt for kp in kps])
            
        # return a tuple of the keypoints and descriptors
        return (kps, descs)
    def match(self, kpsA, featuresA, kpsB, ratio=0.7, minMatches=50):
        # compute the raw matches and initialize the list of actual 
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresB, featuresA, 2)
        matches = []
        
        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each 
            # other
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
                
        # check to see if there are enough matches to process
        if len(matches) > minMatches:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (i, _) in matches])
            ptsB = np.float32([kpsB[j] for (_, j) in matches])
            
            # compute the homography between the two sets of points
            # and compute the ratio of matched points
            (_, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
            
            # return the ratio of the number of matched keypoints
            # to the total number of keypoints
            return float(status.sum()) / status.size
        
        # no matches were found 
        return -1.0