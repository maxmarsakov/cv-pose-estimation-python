"""
pnp detection class
"""
import numpy as np
import cv2 as cv

class pnp_detection:

    def __init__(self, f_x,f_y, c1, c2, method="iterative"):
        """
        params:
        camera_params: intrinsic camera params
        method: iterative

        """
        self.cameraMatrix = np.array([
                [f_x, 0, c1],
                [0, f_y, c2],
                [0, 0, 1],
            ], dtype=np.float32)
        
        self.projection_mat_ = np.zeros((3,4), dtype=np.float64)

    def estimatePoseRansac(self, points_3d_matches, points_2d_matches, \
                confidence,  iterations_count, max_reporjection_error):
        """
        points_3d_matches: 3d points of the matches
        points_2d_matches: 2d poitns of the matches
        confidence: ransac confidence
        iterations_count: number of ransac iterations
        max_reporjection_error: maximum reprojection error
        
        Returns:
        inliers

        Sets: local R,t
        """
        dist_coeffs = np.zeros((4,1), dtype=np.float64)
        r = np.zeros((4,1), dtype=np.float64)
        t = np.zeros((4,1), dtype=np.float64)

        retval, r,t, inliers = cv.solvePnPRansac(points_3d_matches, points_2d_matches, \
            self.cameraMatrix, dist_coeffs, r, t, 
            useExtrinsicGuess=False, iterationsCount=iterations_count, 
            reprojectionError=max_reporjection_error, confidence=confidence )
        R, _ = cv.Rodrigues(r)
        # set 
        self.setProjectionMatrix(R,t)

        print(self.projection_mat_)

        return inliers


    def getRotationTranslation(self):
        """
        returns R and t as
        estimated after pose Ransac
        """
        pass

    
    def setProjectionMatrix(self, R, t):
        """
        sets projection matrix based on R,t
        """
        self.projection_mat_ = np.concatenate([R,t], axis=1)

    def backproject3D(self, point):
        """
        return backprojected 3d point using 
        local Projection matrix
        """
        pass
    
