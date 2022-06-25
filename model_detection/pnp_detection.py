"""
pnp detection class
"""

class pnp_detection:

    def __init__(self, camera_params, method="iterative"):
        """
        params:
        camera_params: intrinsic camera params
        method: iterative

        """
        pass

    def estimatePoseRansac(self, points_3d_matches, points_2d_matches, \
                confidence,  iterations_count, max_reporjection_error):
        """
        points_3d_matches: 3d points of the matches
        points_2d_matches: 2d poitns of the matches
        confidence: ransac confidence
        iterations_count: number of ransac iterations
        max_reporjection_error: maximum reprojection error
        
        Returns:
        inliers, reprojection error

        Sets: local R,t
        """
        pass

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
        pass

    def backproject3D(self, point):
        """
        return backprojected 3d point using 
        local Projection matrix
        """
        pass
    
