"""
model class
for loading textured model
"""
import cv2 as cv
import numpy as np

class Model:

    def __init__(self, points3d, keypoints, descriptors) -> None:

        self.keypoints_ = keypoints
        self.descriptors_ = descriptors
        self.points_3d_ = points3d

    def getKeypoints(self):
        """
        keypoints
        """
        return self.keypoints_

    
    def getDescriptors(self):
        """
        list of descritors of each 3d coordinate
        """
        return self.descriptors_

    def get3DPoints(self):
        """
        list of 3d model coordinates
        """
        return self.points_3d_

    @staticmethod
    def loadModel(filename:str):
        """
        load model from the path
        """
        s = cv.FileStorage(filename, cv.FileStorage_READ)
        points_3d = s.getNode("points_3d").mat()
        points_3d = points_3d.reshape( points_3d.shape[0], points_3d.shape[-1] )
        descriptors = s.getNode("descriptors").mat()
        # parse keypoints
        parsed_keypoints  = np.zeros( (0, 7) , dtype=np.float32)
        if not s.getNode("keypoints").empty():
            keypoints = s.getNode("keypoints")
            m = keypoints.size()
            print("keypoints len",int(m)//7)
            parsed_keypoints = np.zeros( (m) , dtype=np.float32)
            for k in range(m):
                #print(keypoints.at(k).real())
                parsed_keypoints[k] = keypoints.at(k).real()
            parsed_keypoints = parsed_keypoints.reshape(int(m)//7, 7)
        s.release()

        return Model(points_3d, parsed_keypoints, descriptors)