"""
Computer Vision 3D Course Main Project file

This parts including model detection and 3d reconstruction, 
assuming we have registered the model already.
model registration C++ code is available here:
https://github.com/opencv/opencv/blob/4.x/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/src/main_registration.cpp

"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import util
from model_detection import Model
from model_detection import kalman_filter
from model_detection import pnp_detection
from model_detection import robust_matcher
from model_detection import Mesh
import time

def kalman_filter_params(n_states, n_measurements, n_inputs, dt):
    """
    returns predefines kalman filter parameters
    namely, measurement matrix and transition matrix

    params:
    n_states: Number of states
    n_measurements: number of measurements
    n_inputs: number of control actions
    dt: time between measurements

    """
    pass

if __name__ == "__main__":

    print("Started....")
    
    video_source = "./data/test/box.mp4"
    model_path = "./data/test/cookies_ORB.yml"
    mesh_path = "./data/test/box.ply"
    # load the model
    print("Parsing and registering model/mesh....")

    model = Model.loadModel(model_path)
    keypoints_model = model.getKeypoints()
    descriptors_model = model.getDescriptors()
    model_3d_points = model.get3DPoints()
    # load mesh
    mesh = Mesh.loadMesh(mesh_path)

    print("Model/Mesh registration is done")

    # intrinsic camera parameters
    # fx, fy, cx, cy
    camera_params = util.load_camera_parameters("data/calib.npy")

    # init kalman filter
    n_states = 18 # the number of states
    n_measurements = 6 # the number of measured states
    n_inputs = 0 # the number of control actions
    dt = 0.125  #time between measurements (1/FPS)
    kf = kalman_filter( params=kalman_filter_params( n_states, n_measurements, n_inputs, dt ) )

    # init pnp_detection
    # demo parameters
    f = 55
    sx, sy = 22.3, 14.9
    width, height = 640, 480

    pnp = pnp_detection( width*f/sx, height*f/sy, width/2, height/2, method="iterative")
    # est pnp for kalman filter
    pnp_est = pnp_detection(width*f/sx, height*f/sy, width/2, height/2)

    # initalize matcher
    ratio_test = 0.7 # some default value
    # use cross check = True, may provide better alternative to the ration test in D.Lowe SIFT paper
    matcher = robust_matcher( ratio_test=ratio_test, feature_detector="ORB", matcher="BF", use_cross_check=False  )

    # ransac parameters
    ransac_confidence = 0.99
    ransac_iterations = 500
    max_reprojection_error = 6.0 # maximum allowed distance for inlier

    # kalman parameters
    kalman_min_inliers = 30

    # frame loop
    frame_number = 0

    cap = cv.VideoCapture(video_source)
    # for online steaming
    #cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera/no video presented.")
        exit()

    while True:
        start_time = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # step 1 - match the points between the model and the frame
        matches, kp_frame = matcher.fastMatch(frame, keypoints_model, descriptors_model)
    
        # step 2 - 3d-2d correspondencies
        points_2d_matches, points_3d_matches = [], []
        for i in range(len(matches)):
            points_2d_matches.append(kp_frame[ matches[i].queryIdx ].pt)
            points_3d_matches.append(model_3d_points[matches[i].trainIdx])
        # cast to numpy array
        points_2d_matches = np.array(points_2d_matches)
        points_3d_matches = np.array(points_3d_matches)

        # draw outliers
        util.drawPoints( frame, points_2d_matches, color="red")

        cv.imshow('image window', frame)
        # add wait key. window waits until user presses a key
        cv.waitKey(0)
        # and finally destroy/close all open windows
        cv.destroyAllWindows()
        print("matches number", len(matches))

        # at least 4 matches are required for ransac estimation
        if len(matches) >= 4:
            # step 3 - estimate pose of the camera
            inliers = pnp.estimatePoseRansac(  points_3d_matches, points_2d_matches, \
                confidence=ransac_confidence,  iterations_count=ransac_iterations, 
                max_reporjection_error=max_reprojection_error )

            inlier_2d_points = points_2d_matches[ inliers.flatten() ]
            # draw the inliers
            util.drawPoints( frame, inlier_2d_points, color="green" )

            # step 5
            # kalman filter 
            if len(inliers) >= kalman_min_inliers:

                R, t = pnp.getRotationTranslation()
                # update measurements, according to R and t
                measurements = kf.updateMeasurements(R,t)

            # estimate R and t from updated kalman filter
            estimated_R, estimated_t = kf.estimate()

            # step 6 - set estimated projection matrix
            pnp_est.setProjectionMatrix(estimated_R, estimated_t)


        # step 7 - draw pose and coordinate frame
        pose_points2d = []
        l = 5
        pose_points2d.append( pnp_est.backproject3D( (0,0,0) ) ) # axis center
        pose_points2d.append( pnp_est.backproject3D( (l,0,0) ) ) # x axis
        pose_points2d.append( pnp_est.backproject3D( (0,l,0) ) ) # y axis
        pose_points2d.append( pnp_est.backproject3D( (0,0,l) ) ) # z axis

        util.draw3DCoordinateAxes(frame, pose_points2d)

        util.drawObjectMesh(frame, mesh, pnp_est, color="yellow")

        # step BONUS: render some 3d figure on the reconstructed mesh
        # s.t ball rolling
        # TODO

        # DEBUG information
        fps = 1.0 / (time.time() -start_time)
        print("frame number:", frame_number)
        print("fps rate:", fps)
        print("inliers count:", len(inliers))
        print("##################################")

        frame_number += 1

        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break


    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    print("Done")
        






        
        

