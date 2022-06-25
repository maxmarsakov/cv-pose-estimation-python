import numpy as np
import cv2 as cv
import glob

"""
 The purpose of this code is to create the calibration matrix
 In other words, to find the internal camera parameters and save it
 It outpoots the total reporojection error as well
 The calibration matrix ma be used further in 3d reconstruction
"""
if __name__ == "__main__":

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    GRID_SIZE_0 = 6
    GRID_SIZE_1 = 9

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((GRID_SIZE_0*GRID_SIZE_1,3), np.float32)
    objp[:,:2] = np.mgrid[0:GRID_SIZE_0,0:GRID_SIZE_1].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('calibrate/*.jpeg')
    for fname in images:
        print(fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (GRID_SIZE_0,GRID_SIZE_1), cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            print("yes")
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (GRID_SIZE_1,GRID_SIZE_0), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
    cv.destroyAllWindows()

    # calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(mtx)

    # save the calibration matrix
    np.save("K.npy", mtx)

    # print the reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )