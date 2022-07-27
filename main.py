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

    parser.add_argument('-video', '--source_video', type=str, help='')
    parser.add_argument('-model', '--model_path', type=str, help='')
    parser.add_argument('-mesh', '--mesh_path', type=str, help='')
    parser.add_argument('-k', '--use_kalman', action="store_true", help='')
    parser.add_argument('-in', '--kalman_inliers',  type=int, help='')   
    parser.add_argument('-d', '--detector',  type=str, choices=["ORB","SIFT"], help='')   
    parser.add_argument('-m', '--matcher',  type=str, choices=["BF", "FLANN"], help='')   
    parser.add_argument('-c', '--ransac_confidence',  type=float,  help='')   
    parser.add_argument('-it', '--ransac_iterations',  type=int,  help='')   
    parser.add_argument('-e', '--reprojection_error',  type=float,  help='')   
    parser.add_argument('-v', '--verbose', action="store_true",  help='')   
    parser.add_argument('-w','--webcam', action="store_true", help='')
    parser.add_argument('-r','--render', action="store_true", help='')
    parser.add_argument('-o','--output',type=str, help='output video')

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

    kalman_min_inliers=0
    if useKalmanFilter:
        n_states = 18 # the number of states
        n_measurements = 6 # the number of measured states
        n_inputs = 0 # the number of control actions
        dt = 0.5  #time between measurements (1/FPS) # 0.125
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
    matcher_name = args.matcher if args.matcher else "BF"

    matcher = robust_matcher( ratio_test=ratio_test, feature_detector=detector, 
        nfeatures=num_detected_points, matcher=matcher_name, use_cross_check=False  )

    # ransac parameters
    ransac_confidence = args.ransac_confidence if args.ransac_confidence else 0.99 # to change
    ransac_iterations = args.ransac_iterations if args.ransac_iterations else 500
    # increasing this parameter made most significance for the results
    max_reprojection_error = args.reprojection_error if args.reprojection_error else 20.0 # maximum allowed distance for inlier

    renderObject = args.render if args.render else False # to render speical object?
    # frame loop
    frame_number = 0
    # store R
    prev_R = np.zeros((3,3), dtype=np.float64)

    video_name = 'project.avi' if not args.output else args.output
    out_video = cv.VideoWriter(video_name,cv.VideoWriter_fourcc(*'DIVX'), 15, (640,480))

    print("Args:")
    print("source_video:",video_source)
    print("model_path:",model_path)
    print("mesh_path:",mesh_path)
    print("use_kalman:",useKalmanFilter)
    print("kalman_inliers:",kalman_min_inliers)
    print("detector:",detector)
    print("matcher:",matcher_name)
    print("ransac_confidence:",ransac_confidence)
    print("ransac_iterations:",ransac_iterations)
    print("reprojection_error:",max_reprojection_error)
    print("verbose:",args.verbose)
    print("is webcam:", args.webcam is not None)
    print("render:", args.render is not None)
    print("video:", video_name)

    if args.webcam:
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture(video_source)
    

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

        if args.verbose:
            print("matches number", len(matches))

        # at least 4 matches are required for ransac estimation
        retval=False
        if len(matches) >= 4:
            # step 3 - estimate pose of the camera
            retval, inliers = pnp.estimatePoseRansac(  points_3d_matches, points_2d_matches, \
                confidence=ransac_confidence,  iterations_count=ransac_iterations, 
                max_reprojection_error=max_reprojection_error )

        if not retval:
            if args.verbose:
                print("ransac failed")
            continue

        if inliers is not None:
            inlier_2d_points = points_2d_matches[ inliers.flatten() ]
            # draw the inliers
            util.drawPoints( frame, inlier_2d_points, color="green" )

       
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

        # red - X
        # blue - Y
        # green - Z
        if len(inliers)>=5:
            util.draw3DCoordinateAxes(frame, pose_points2d)

            util.drawObjectMesh(frame, mesh.triangles_, mesh.vertices_, pnp_est, color="yellow")

            # step 8: render some 3d figure on the reconstructed mesh
            # pyramidic roof
            if renderObject:
                util.drawObjectTrianglesCountour(frame, roof_mesh, roof_vertices, pnp_est, colors=["red", "blue", "green", "yellow"])

        # DEBUG information
        fps = 1.0 / (time.time() -start_time)
        # fps
        cv.putText(frame,f'FPS: {int(fps)}', (50,50), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1,2)

        if args.verbose:
            print("frame number:", frame_number)
            print("fps rate:", fps)
            if inliers is not None:
                print("inliers count:", len(inliers))
            print("##################################")

        frame_number += 1

        out_video.write(frame)

        # Display the resulting frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    #save video
    out_video.release()

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    print("Done")

        
        

