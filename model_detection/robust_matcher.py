"""
robust matcher class
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

class robust_matcher:

    def __init__(self, ratio_test:float = 0.75, feature_detector:str = "orb", nfeatures:int = 2000, matcher:str="BF", use_cross_check:bool=False):
        """
        params:
        ratio_test: float
        feature_detector: "orb"
        matcher: either Brute Force or Flann
        use_cross_check: bool parameter for better results using BFmatcher
        """
        assert matcher == "BF" or matcher == "FLANN"
        assert feature_detector == "ORB" or feature_detector == "SIFT"

        # feature detector
        self.ratio_test_ = ratio_test
        if feature_detector == "ORB":
            # set detector
            self.feature_detector_ = cv.ORB_create(nfeatures=nfeatures)
            # set params for matcher
            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

        elif feature_detector == "SIFT":
            self.feature_detector_ = cv.SIFT_create(nfeatures=nfeatures)
            # params for the matcher
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)


        # setting matcher
        if matcher == "BF":
            norm = cv.NORM_L2
            if feature_detector == "SIFT":
                norm = cv.NORM_L2
            elif feature_detector == "ORB":
                # since we are using ORB descriptors, the distance will be measure 
                # through hamming norm
                norm = cv.NORM_HAMMING

            self.matcher_ = cv.BFMatcher(norm, crossCheck=use_cross_check)
        elif matcher == "FLANN":
            search_params = dict(checks=50)   # how much the number of trees sholud the index recuresively passed
            
            self.matcher_ = cv.FlannBasedMatcher(index_params,search_params)
        
        self.use_cross_check_ = use_cross_check

    def computeKeyPoints_(self, frame):
        """
        returns key points from frame and the corresponding descroptors,
        based on self.feature_detector_
        returns: (keypoints, descriptors)
        """
        return self.feature_detector_.detectAndCompute(frame, None)

    def matchDescriptors_(self, des1 ,des2):
        """
        matching between the two descriptors,
        based on self.matcher_
        """
        #print(des1.shape,des2.shape)
        if self.use_cross_check_:
            matches  = self.matcher_.match(des1, des2)
            # sort the matches
            matches = sorted(matches, key = lambda x:x.distance)
            return matches

        matches = self.matcher_.knnMatch(des1,des2,k=2)
        return matches
        
    def robustMatch(self,frame, keypoints_model, model_des):
        """
        not fast match 
        """
        pass

    def fastMatch(self, frame, keypoints_model, model_des):
        """
        params:
        frame: image
        keypoints_model: given keypoints model
        descriptors_model: given descriptors of the model

        match between the frame and model keypoints using orb 
        returns keypoints from the scene and matched points
        """
        # compute key points & descriptors from the frame
        frame_kp, frame_des = self.computeKeyPoints_(frame)
        #print("frame keypoints",len(frame_kp))

        # match two image descriptors
        matches = self.matchDescriptors_(frame_des, model_des)

        if self.use_cross_check_:
            return matches, frame_kp
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < self.ratio_test_ * n.distance:
                good.append(m)

        # return good matches
        return good, frame_kp
