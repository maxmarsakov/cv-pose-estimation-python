"""
kalman filter class
"""
import numpy as np
import cv2 as cv
import util

class kalman_filter:

    def __init__(self, n_states, n_measurements, n_inputs, dt) -> None:
        """
        params: 
        params - kalman filter state etc
        """

        # initialize kalman filter
        self.n_states_ = n_states
        self.n_measurements_ = n_measurements
        self.n_inputs_ = n_inputs
        self.dt_ = dt

        self.kf_ = cv.KalmanFilter(n_states, n_measurements, n_inputs, type=cv.CV_64FC1)

        self.prev_measurements_ = np.zeros( (self.n_measurements_) , dtype=np.float64)

        # set error covariance matrices
        self.kf_.processNoiseCov = cv.setIdentity(self.kf_.processNoiseCov, 1e-5) # process noise
        self.kf_.measurementNoiseCov = cv.setIdentity(self.kf_.measurementNoiseCov, 1e-2) # measuerement noise
        self.kf_.errorCovPost = cv.setIdentity(self.kf_.errorCovPost, 1) # error covariance

    def setMeasurementMatrix(self, mat: np.ndarray):
        """
        helper function for setting measurement matrix
        """
        self.kf_.measurementMatrix = mat

    def setTransitionMatrix(self, mat: np.ndarray):
        """
        helper function for setting transition matrix
        """
        self.kf_.transitionMatrix = mat

    def updateMeasurements(self, prev=False, R: np.ndarray = None, t: np.ndarray = None ):
        """
        params:
        given measurement update observed R,t
        prev: boolean. if True will use previous recorder measurements
        based on prediction correction kalman filter
        scheme
        """
        if not prev:
            # convert to local measurements
            measurements = self.fillMeasurements(R, t)
            # store prev measurements
            self.prev_measurements_ = measurements
        else:
            #return
            measurements = self.prev_measurements_

        #print("process noise cov", self.kf_.processNoiseCov)
        #print("measurement noise cov", self.kf_.measurementNoiseCov)
        #print("process errorCovPost cov", self.kf_.errorCovPost)
        
        # predict
        predicted_mat = self.kf_.predict()
        # correct phase
        estimated_mat = self.kf_.correct(measurements)

        # estimate translation
        t = np.zeros( (3,), dtype=np.float64)
        t[0] = estimated_mat[0]
        t[1] = estimated_mat[1]
        t[2] = estimated_mat[2]

        # estimated euler angles
        quaterenion = np.zeros( (3,), dtype=np.float64)
        quaterenion[0] = estimated_mat[9]
        quaterenion[1] = estimated_mat[10]
        quaterenion[2] = estimated_mat[11]

        # convert estimated quaterenion to R
        R = util.euler2rot(quaterenion)

        self.R_ = R
        self.t_ = t

    def fillMeasurements(self, R, t):
        """
        given measured R and t, return 
        measurements np array which will be used in kalman filter
        """
        measurements = np.zeros( (self.n_measurements_) , dtype=np.float64)

        quateranion = util.rot2euler(R)

        measurements[0] = t[0]
        measurements[1] = t[1]
        measurements[2] = t[2]
        measurements[3] = quateranion[0]
        measurements[4] = quateranion[1]
        measurements[5] = quateranion[2]

        return measurements
        

    def estimate(self):
        """
        return estimated R,t based on measurements
        and parameters
        """
        return self.R_, self.t_