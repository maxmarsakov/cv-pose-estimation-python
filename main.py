"""
Computer Vision 3D Course Main Project file

This parts including model detection and 3d reconstruction.
@authors: 
2022

"""
import argparse
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

def parseArgs():
    parser = argparse.ArgumentParser(
        description='Welcome to the interactive 3D model reconstruction based on textured object')
    parser.add_argument('-n', '--npoints', type=int, help="number of Keypoints")

    parser.add_argument('-source_video', '--source_video', type=str, help='')
    parser.add_argument('-model_path', '--model_path', type=str, help='')
    parser.add_argument('-mesh_path', '--mesh_path', type=str, help='')
    parser.add_argument('-kalman', '--use_kalman', action="store_true", help='')
    parser.add_argument('-kalman_inliers', '--kalman_inliers',  type=int, help='')   
    parser.add_argument('-detector', '--detector',  type=str, choices=["ORB"], help='')   
    parser.add_argument('-matcher', '--matcher',  type=str, choices=["BF", "FLANN"], help='')   
    parser.add_argument('-confidence', '--ransac_confidence',  type=float,  help='')   
    parser.add_argument('-i', '--ransac_iterations',  type=int,  help='')   
    parser.add_argument('-repr', '--reprojection_error',  type=float,  help='')   


    #parser.add_argument('-t', '--train', action='store_true', help='Train the AI')
    return parser.parse_args()

def init_kalman_filter(n_states, n_measurements, n_inputs, dt):
    """
    returns predefines kalman filter parameters
    namely, measurement matrix and transition matrix

    params:
    n_states: Number of states
    n_measurements: number of measurements
    n_inputs: number of control actions
    dt: time between measurements

    returns: instance of kalman filter
    """
    kf = kalman_filter(n_states, n_measurements, n_inputs, dt)
    # speed
    dt2 = 0.5 * (dt**2)
    
    # n_states X n_states
    kf.setTransitionMatrix(np.array([ 
        [1, 0, 0, dt, 0, 0, dt2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, dt, 0, 0, dt2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, dt, 0, 0, dt2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, dt2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, dt2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, dt2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=np.float64))

    # n_measurements X n_states
    kf.setMeasurementMatrix(np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    ],dtype=np.float64))

    return kf
    

if __name__ == "__main__":

    args = parseArgs()

    print("Started....")
    
    video_source = args.source_video if args.source_video else "./data/test/box.mp4"
    model_path = args.model_path if args.model_path else "./data/test/cookies_ORB.yml"
    mesh_path = args.mesh_path if args.mesh_path else "./data/test/box.ply"
    # load the model
    print("Parsing and registering model/mesh....")

    model = Model.loadModel(model_path)
    keypoints_model = model.getKeypoints()
    descriptors_model = model.getDescriptors()
    model_3d_points = model.get3DPoints()
    # load mesh
    mesh = Mesh.loadMesh(mesh_path)
    # load *roof* mesh
    roof_mesh, roof_vertices = mesh.loadRoofMesh()

    print("Model/Mesh registration is done")

    # intrinsic camera parameters
    # fx, fy, cx, cy
    camera_params = util.load_camera_parameters("data/calib.npy")

    # init kalman filter
    useKalmanFilter = args.use_kalman if args.use_kalman else False

    if useKalmanFilter:
        n_states = 18 # the number of states
        n_measurements = 6 # the number of measured states
        n_inputs = 0 # the number of control actions
        dt = 0.125  #time between measurements (1/FPS) # 0.125
        # minimal number of inliers required for kalman filter
        kalman_min_inliers = args.kalman_inliers if args.kalman_inliers else 50
        kf = init_kalman_filter( n_states, n_measurements, n_inputs, dt )

    # init pnp_detection
    # demo parameters
    f = 55
    sx, sy = 22.3, 14.9
    width, height = 640, 480

    pnp = pnp_detection( width*f/sx, height*f/sy, width/2, height/2, method="iterative")
    # est pnp for kalman filter
    pnp_est = pnp_detection(width*f/sx, height*f/sy, width/2, height/2)

    # initalize matcher
    ratio_test = 0.70 # default value was 0.7, changed to 0.9 for better results
    # use cross check = True, may provide better alternative to the ration test in D.Lowe SIFT paper
    num_detected_points = args.npoints if args.npoints else 2000
    detector = args.detector if args.detector else "ORB"
    matcher = args.matcher if args.matcher else "BF"

    matcher = robust_matcher( ratio_test=ratio_test, feature_detector="ORB", 
        nfeatures=num_detected_points, matcher="BF", use_cross_check=False  )

    # ransac parameters
    ransac_confidence = args.ransac_confidence if args.ransac_confidence else 0.99 # to change
    ransac_iterations = args.ransac_iterations if args.ransac_iterations else 500
    # increasing this parameter made most significance for the results
    max_reprojection_error = args.reprojection_error if args.reprojection_error else 20.0 # maximum allowed distance for inlier

    renderObject = True # to render speical object?
    # frame loop
    frame_number = 0
    # store R
    prev_R = np.zeros((3,3), dtype=np.float64)

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
            points_3d_matches.append(model_3d_points[ matches[i].trainIdx ])
        # cast to numpy array
        points_2d_matches = np.array(points_2d_matches)
        points_3d_matches = np.array(points_3d_matches)

        # draw outliers
        util.drawPoints( frame, points_2d_matches, color="red")
        # is measurement good for kalman
        good_measurement = False

        print("matches number", len(matches))

        # at least 4 matches are required for ransac estimation
        if len(matches) >= 4:
            # step 3 - estimate pose of the camera
            inliers = pnp.estimatePoseRansac(  points_3d_matches, points_2d_matches, \
                confidence=ransac_confidence,  iterations_count=ransac_iterations, 
                max_reprojection_error=max_reprojection_error )
            
            if inliers is not None:
                inlier_2d_points = points_2d_matches[ inliers.flatten() ]
                # draw the inliers
                util.drawPoints( frame, inlier_2d_points, color="green" )

            #time.sleep(3)
           
            # step 5 - KF
            R, t = pnp.getRotationTranslation()

            # if number of inliers of kalman is, update measurements\
            if useKalmanFilter:
                if inliers is not None and len(inliers) >= kalman_min_inliers:
                    good_measurement = True
                    # update measurements in kalman filter, according to R and t

                    kf.updateMeasurements(R=R,t=t)
                else:
                    # else estimate using previous measurements
                    kf.updateMeasurements(prev=True)

                # estimate R and t from updated kalman filter
                estimated_R, estimated_t = kf.estimate()
                # step 6 - set estimated projection matrix
                pnp_est.setProjectionMatrix(estimated_R, estimated_t)
            else:
                pnp_est.setProjectionMatrix(R, t)


        # step 7 - draw pose and coordinate frame
        #if good_measurement:
        pose_points2d = []
        l = 5
        pose_points2d.append( pnp_est.backproject3D( (0,0,0) ) ) # axis center
        pose_points2d.append( pnp_est.backproject3D( (l,0,0) ) ) # x axis
        pose_points2d.append( pnp_est.backproject3D( (0,l,0) ) ) # y axis
        pose_points2d.append( pnp_est.backproject3D( (0,0,l) ) ) # z axis

        print("drawing object mesh and coordinated axis")
        # red - X
        # blue - Y
        # green - Z

        util.draw3DCoordinateAxes(frame, pose_points2d)

        util.drawObjectMesh(frame, mesh.triangles_, mesh.vertices_, pnp_est, color="yellow")

        # step 8: render some 3d figure on the reconstructed mesh
        # pyramidic roof
        if renderObject:
            util.drawObjectTrianglesCountour(frame, roof_mesh, roof_vertices, pnp_est, colors=["red", "blue", "green", "yellow"])

        # DEBUG information
        fps = 1.0 / (time.time() -start_time)
        print("frame number:", frame_number)
        print("fps rate:", fps)
        if inliers is not None:
            print("inliers count:", len(inliers))
        print("##################################")

        frame_number += 1

        # Our operations on the frame come here
        # Display the resulting frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    print("Done")

        
        

