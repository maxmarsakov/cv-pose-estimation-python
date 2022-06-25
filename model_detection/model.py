"""
model class
for loading textured model
"""

class Model:

    def __init__(self, keypoints, descriptors) -> None:

        self.keypoints_ = keypoints
        self.descriptors_ = descriptors

    def getKeypoints(self):

        pass

    
    def getDescriptors(self):
        """
        list of descritors of each 3d coordinate
        """
        pass

    def get3DPoints(self):
        """
        list of 3d model coordinates
        """
        pass

    @staticmethod
    def loadModel(path:str):
        """
        load model from the path
        TODO
        """
        keypoints = None
        descriptors = None

        return Model(keypoints, descriptors)
    
