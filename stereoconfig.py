import numpy as np
import cv2

#双目相机参数
class stereoCamera(object):
    def __init__(self):

        #左相机内参数
        self.cam_matrix_left = np.array([[3.13229149e+03, 0., 1.38078252e+03], [0., 3.11524597e+03, 1.25386237e+03], [0., 0., 1.]])
        #右相机内参数
        self.cam_matrix_right = np.array([[3.13229149e+03, 0., 1.38078252e+03], [0., 3.11524597e+03, 1.25386237e+03], [0., 0., 1.]])

        #左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[ 0.04632877,0.13619159 -0.0259212,0.00131995,0.72588067]])
        self.distortion_r = np.array([[-0.03348, 0.08901, -0.00327, 0.00330, 0.00000]])

        #旋转矩阵
        om = np.array([-0.22254491,0.22353666,1.54739114])
        self.R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
        #平移矩阵
        self.T = np.array([0.50262619,-0.70639708, 2.62975255])

        # 主点列坐标的差
        self.doffs = 0.0

        # 指示上述内外参是否为经过立体校正后的结果
        self.isRectified = False

    def setMiddleBurryParams(self):
        self.cam_matrix_left = np.array([[3997.684, 0, 225.0],
                                         [0., 3997.684, 187.5],
                                         [0., 0., 1.]])
        self.cam_matrix_right = np.array([[3997.684, 0, 225.0],
                                          [0., 3997.684, 187.5],
                                          [0., 0., 1.]])
        self.distortion_l = np.zeros(shape=(5, 1), dtype=np.float64)
        self.distortion_r = np.zeros(shape=(5, 1), dtype=np.float64)
        self.R = np.identity(3, dtype=np.float64)
        self.T = np.array([[-193.001], [0.0], [0.0]])
        self.doffs = 131.111
        self.isRectified = True

