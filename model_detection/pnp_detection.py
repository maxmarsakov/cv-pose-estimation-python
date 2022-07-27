"""
pnp detection class
"""
from turtle import shape
import numpy as np
import cv2 as cv
import time

FIX=True

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
            ], dtype=np.float64)
        
        self.projection_mat_ = np.zeros((3,4), dtype=np.float64)

        self.R_, self.t_ = None, None
        # save precomputed values of r and t to
        # use them in the extrinsic gues
        self.pre_set_ = False

        self.pre_t_, self.pre_r_ = None, None

    def estimatePoseRansac(self, points_3d_matches, points_2d_matches, \
                confidence,  iterations_count, max_reprojection_error):
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

        useExstrinsic:bool = False # if set to true, will use prer and pret
        # as initial guesss
        if not self.pre_set_:
            self.pre_set_ = True
            useExstrinsic = False
            self.pre_t_ = np.zeros((3,1), dtype=np.float64)
            self.pre_r_ = np.zeros((3,1), dtype=np.float64)

        flags = cv.SOLVEPNP_EPNP

        retval, r,t, inliers = cv.solvePnPRansac(points_3d_matches, points_2d_matches, \
            self.cameraMatrix, dist_coeffs, self.pre_r_, self.pre_t_, 
            useExtrinsicGuess=useExstrinsic, iterationsCount=iterations_count, 
            reprojectionError=max_reprojection_error, confidence=confidence, flags=flags )

        #print(r, t)
        # some what when signs of the output rotation are flipped
        # between r[0] an r[1], produced wrong results
        if FIX:
            if r[1] > 0:
                # flip the values again
                r[1] = -1*r[1]
                r[0] = -1*r[0]
                #time.sleep(1)
            if t[1] < 0: # another fix
                t[1] = -1 * t[1]

        self.pre_r_, self.pre_t_ = r, t

        R, _ = cv.Rodrigues(r)
        
        self.setProjectionMatrix(R,t)

        self.R_ = R
        self.t_ = t

        return retval, inliers


    def getRotationTranslation(self):
        """
        returns R and t as
        estimated after pose Ransac
        """
        return self.R_, self.t_

    
    def setProjectionMatrix(self, R, t):
        """
        sets projection matrix based on R,t
        """
        
        self.projection_mat_ = np.concatenate([R,t.reshape(3,1)], axis=1)

    def backproject3D(self, point):
        """
        return backprojected 3d point using 
        local Projection matrix
        """
        p = np.array(point).reshape(3)
        p = np.hstack([point,np.ones(1)]).reshape(4,1)


        point_2d = self.cameraMatrix @ self.projection_mat_ @ p

        last_point=(point_2d[2] if point_2d[2] != 0 else 1.0)
        return (int(point_2d[0]/last_point), int(point_2d[1]/last_point))

    
