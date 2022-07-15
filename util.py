"""
util function go here
"""
import cv2 as cv
from model_detection import pnp_detection
from model_detection import Mesh
import numpy as np
import math

COLORS = {
    "red": (0,0,255),
    "green": (0,255,0),
    "blue": (255,0,0),
    "yellow": (17,223,233),
    "black": (0,0,0)
}
RADIUS=4
LINE_TYPE=8
FONT_SCALE=0.75
THICKNESS_FONT=2

def load_video(path:str):
    """
    loads video path in form of frames
    """    
    pass


def load_model(path:str):

    pass

def load_mesh(path:str):
    """
    loads mesh
    """  
    pass

def load_camera_parameters(path:str):

    pass

def drawPoints(frame, points_2d, color:str):
    """
    draws points on the frame image
    """  
    for point in points_2d:
        cv.circle(frame, (int(point[0]),int(point[1])), radius=RADIUS, color=COLORS[color], thickness=-1 )

def draw2DPoints(frame, points_2d, points_3d, color:str):
    """
    draws 2d points and their labels
    NOT necessary
    """
    pass

def drawArrow(frame, p, q, color, arrowMagnitude, thickness):
    """
    draws single arrow
    """
    cv.line(frame, p, q, color=COLORS[color], thickness=thickness, lineType=LINE_TYPE )

    # TODO - draw the two arrow segments
    """
    const double PI = CV_PI;
    //compute the angle alpha
    double angle = atan2((double)p.y-q.y, (double)p.x-q.x);
    //compute the coordinates of the first segment
    p.x = (int) ( q.x +  arrowMagnitude * cos(angle + PI/4));
    p.y = (int) ( q.y +  arrowMagnitude * sin(angle + PI/4));
    //Draw the first segment
    cv::line(image, p, q, color, thickness, line_type, shift);
    //compute the coordinates of the second segment
    p.x = (int) ( q.x +  arrowMagnitude * cos(angle - PI/4));
    p.y = (int) ( q.y +  arrowMagnitude * sin(angle - PI/4));
    //Draw the second segment
    cv::line(image, p, q, color, thickness, line_type, shift);
    """


def draw3DCoordinateAxes(frame, list_points2d):
    """
    draws axis from the list on the frame
    """ 
    origin = list_points2d[0]
    pointX = list_points2d[1]
    pointY = list_points2d[2]
    pointsZ = list_points2d[3]

    drawArrow(frame, origin, pointX, "red", 9, 2)
    drawArrow(frame, origin, pointY, "blue", 9, 2)
    drawArrow(frame, origin, pointsZ, "green", 9, 2)

    cv.circle(frame, origin, radius=RADIUS//2, color=COLORS["black"], thickness=-1)

def drawObjectTrianglesCountour(frame, triangles, vertices, pnp_est, colors=[]):
    """
    instead of drawing simple object mesh, we draw
    contour filled triangles
    """
    for i,triangle in enumerate(triangles):
        #if colors[i] == "red":
            # only one is visible at a time
            #continue
        # backproject 3d points
        # TODO - bulk pnp backprojection
        vertex1 = vertices[triangle[0]]
        vertex2 = vertices[triangle[1]]
        vertex3 = vertices[triangle[2]]

        point_2d_0 = pnp_est.backproject3D( vertex1 )
        point_2d_1 = pnp_est.backproject3D( vertex2 )
        point_2d_2 = pnp_est.backproject3D( vertex3 )

        triangle_cnt = np.array( [point_2d_0, point_2d_1, point_2d_2] )

        # draw lines
        cv.drawContours(frame, [triangle_cnt], 0, COLORS[colors[i]], -1)

def drawObjectMesh(frame, triangles, vertices, pnp_est: pnp_detection, color="yellow", putText=False):
    """
    using object mesh and pnp estimate instance
    renders the mesh on the frame
    - putText: True if display the text annotations near all the vertices
    """ 

    if putText:
        for vertex in vertices:
            # draw annotation of the3d model points
            point_2d = pnp_est.backproject3D( vertex )
            cv.putText(frame, str(vertex), point_2d,fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.4, color=COLORS["blue"], thickness=1, lineType=cv.LINE_AA)

    for triangle in triangles:
        # backproject 3d points
        # TODO - bulk pnp backprojection
        vertex1 = vertices[triangle[0]]
        vertex2 = vertices[triangle[1]]
        vertex3 = vertices[triangle[2]]

        point_2d_0 = pnp_est.backproject3D( vertex1 )
        point_2d_1 = pnp_est.backproject3D( vertex2 )
        point_2d_2 = pnp_est.backproject3D( vertex3 )
        #print("done")

        # draw lines
        cv.line(frame, point_2d_0, point_2d_1, COLORS[color], 1)
        cv.line(frame, point_2d_1, point_2d_2, COLORS[color], 1)
        cv.line(frame, point_2d_2, point_2d_0, COLORS[color], 1)

"""
misc
"""
def get_translation_error(t_true: np.ndarray, t: np.ndarray):
    return cv.norm(t_true, t)

def get_rotation_error(R_true: np.ndarray, R: np.ndarray):

    return cv.norm(cv.Rodrigues(-R_true @ R.T))

# Calculates Rotation Matrix given euler angles.
# https://learnopencv.com/rotation-matrix-to-euler-angles/
def euler2rot(theta):
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rot2euler(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])