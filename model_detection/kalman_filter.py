"""
kalman filter class
"""

class kalman_filter:

    def __init__(self, params) -> None:
        """
        params: 
        params - kalman filter state etc
        """
        self.R_ = None
        self.t_ = None    

    def updateMeasurements(self, R,t):
        """
        params:
        given R,t update local measurments
        """
        self.R_ = R
        self.t_ = t

    def estimate(self):
        """
        return estimated R,t based on measurements
        and parameters
        """
        return self.R_, self.t_