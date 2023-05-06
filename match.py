# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
#
# def SURF(img):
#     surf = cv2.xfeatures2d.SURF_create()
#     kp, des = surf.detectAndCompute(img, None)
#     cv2.drawKeypoints(img, kp, img, (0, 255, 0))
#
#     # 图像显示
#     plt.figure(figsize=(10, 8), dpi=100)
#     plt.imshow(img[:, :, ::-1])
#     plt.xticks([]), plt.yticks([])
#     plt.show()
#     return kp, des
#
#
# def ByBFMatcher(img1, img2, kp1, kp2, des1, des2, flag="ORB"):
#     """
#     （1）暴力法
#     :param img1: 匹配图像1
#     :param img2: 匹配图像2
#     :param kp1: 匹配图像1的特征点
#     :param kp2: 匹配图像2的特征点
#     :param des1: 匹配图像1的描述子
#     :param des2: 匹配图像2的描述子
#     :return:
#     """
#     if (flag == "SIFT" or flag == "sift"):
#         # SIFT方法或SURF
#         bf = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)
#     else:
#         # ORB方法
#         bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
#     ms = bf.match(des1, des2)
#     # ms = sorted(ms, key=lambda x: x.distance)
#     img3 = cv2.drawMatches(img1, kp1, img2, kp2, ms, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     cv2.imwrite("1.jpg", img3)
#     # cv2.imshow("Matches", img3)
#     plt.imshow(img3, ), plt.show()
#     cv2.waitKey(0)
#     return ms
#
#
# img1 = cv2.imread("1.jpg")
# img2 = cv2.imread("2.jpg")
# kp1, des1 = SURF(img1)
# kp2, des2 = SURF(img2)
# matches = ByBFMatcher(img1, img2, kp1, kp2, des1, des2, "SIFT")




#Harris算法
# from pylab import *
# from PIL import Image
# from PCV.localdescriptors import harris
# from PCV.tools.imtools import imresize
#
# im1 = array(Image.open("1.jpg").convert("L"))
# im2 = array(Image.open("2.jpg").convert("L"))
#
# wid = 5
# harrisim = harris.compute_harris_response(im1, 5)
# filtered_coords1 = harris.get_harris_points(harrisim, wid+1)
# d1 = harris.get_descriptors(im1, filtered_coords1, wid)
#
# harrisim = harris.compute_harris_response(im2, 5)
# filtered_coords2 = harris.get_harris_points(harrisim, wid+1)
# d2 = harris.get_descriptors(im2, filtered_coords2, wid)
#
# print ('starting matching')
# matches = harris.match_twosided(d1, d2)
#
# figure()
# gray()
# harris.plot_matches(im1, im2, filtered_coords1, filtered_coords2, matches)
# show()


# """
# 图像特征点的检测与匹配
# 主要涉及：
# 1、ORB
# 2、SIFT
# 3、SURF
# """
#
# """
# 一、图像特征点的检测
# """
import numpy as np
import cv2
from matplotlib import pyplot as plt


# def ORB(img):
#     """
#      ORB角点检测
#      实例化ORB对象
#     """
#     orb = cv2.ORB_create(nfeatures=500)
#     """检测关键点，计算特征描述符"""
#     kp, des = orb.detectAndCompute(img, None)
#
#     # 将关键点绘制在图像上
#     img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
#
#     # 画图
#     plt.figure(figsize=(10, 8), dpi=100)
#     plt.imshow(img2[:, :, ::-1])
#     plt.xticks([]), plt.yticks([])
#     plt.show()
#     return kp, des
#
#
# def SIFT(img):
#     # SIFT算法关键点检测
#     # 读取图像
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # SIFT关键点检测
#     # 1. 实例化sift
#     sift = cv2.xfeatures2d.SIFT_create()
#
#     # 2. 利用sift.detectAndCompute()检测关键点并计算
#     kp, des = sift.detectAndCompute(gray, None)
#     # gray: 进行关键带你检测的图像，注意是灰度图像
#     # kp: 关键点信息，包括位置，尺度，方向信息
#     # des: 关键点描述符，每个关键点对应128个梯度信息的特征向量
#
#     # 3. 将关键点检测结果绘制在图像上
#     # cv2.drawKeypoints(image, keypoints, outputimage, color, flags)
#     # image: 原始图像
#     # keypoints: 关键点信息，将其绘制在图像上
#     # outputimage: 输出图片，可以是原始图像
#     # color: 颜色设置，通过修改(b, g, r)的值，更改画笔的颜色，b = 蓝色, g = 绿色, r = 红色
#     # flags: 绘图功能的标识设置
#     # 1. cv2.DRAW_MATCHES_FLAGS_DEFAULT: 创建输出图像矩阵，使用现存的输出图像绘制匹配对象和特征点，对每一个关键点只绘制中间点
#     # 2. cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG: 不创建输出图像矩阵，而是在输出图像上绘制匹配对
#     # 3. cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: 对每一个特征点绘制带大小和方向的关键点图形
#     # 4. cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS: 单点的特征点不被绘制
#     cv2.drawKeypoints(img, kp, img, (0, 255, 0))
#
#     # 图像显示
#     plt.figure(figsize=(10, 8), dpi=100)
#     plt.imshow(img[:, :, ::-1])
#     plt.xticks([]), plt.yticks([])
#     plt.show()
#     return kp, des
#
#
# def SURF(img):
#     surf = cv2.xfeatures2d.SURF_create()
#     kp, des = surf.detectAndCompute(img, None)
#     cv2.drawKeypoints(img, kp, img, (0, 255, 0))
#
#     # 图像显示
#     plt.figure(figsize=(10, 8), dpi=100)
#     plt.imshow(img[:, :, ::-1])
#     plt.xticks([]), plt.yticks([])
#     plt.show()
#     return kp, des
#
#
# """
# 2.图像特征点匹配方法
# （1）暴力法
# （2）FLANN匹配器
# """
#
#
# def ByBFMatcher(img1, img2, kp1, kp2, des1, des2, flag="ORB"):
#     """
#     （1）暴力法
#     :param img1: 匹配图像1
#     :param img2: 匹配图像2
#     :param kp1: 匹配图像1的特征点
#     :param kp2: 匹配图像2的特征点
#     :param des1: 匹配图像1的描述子
#     :param des2: 匹配图像2的描述子
#     :return:
#     """
#     if (flag == "SIFT" or flag == "sift"):
#         # SIFT方法
#         bf = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)
#     else:
#         # ORB方法
#         bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
#     ms = bf.knnMatch(des1, des2, k=2)
#     # ms = sorted(ms, key=lambda x: x.distance)
#     # img3 = cv2.drawMatches(img1, kp1, img2, kp2, ms, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     # cv2.imshow("Matches", img3)
#     # cv2.waitKey(0)
#     return ms
#
#
# def ByFlann(img1, img2, kp1, kp2, des1, des2, flag="ORB"):
#     """
#         （1）FLANN匹配器
#         :param img1: 匹配图像1
#         :param img2: 匹配图像2
#         :param kp1: 匹配图像1的特征点
#         :param kp2: 匹配图像2的特征点
#         :param des1: 匹配图像1的描述子
#         :param des2: 匹配图像2的描述子
#         :return:
#         """
#     if (flag == "SIFT" or flag == "sift"):
#         # SIFT方法
#         FLANN_INDEX_KDTREE = 1
#         index_params = dict(algorithm=FLANN_INDEX_KDTREE,
#                             trees=5)
#         search_params = dict(check=50)
#     else:
#         # ORB方法
#         FLANN_INDEX_LSH = 6
#         index_params = dict(algorithm=FLANN_INDEX_LSH,
#                             table_number=6,
#                             key_size=12,
#                             multi_probe_level=1)
#         search_params = dict(check=50)
#     # 定义FLANN参数
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(des1, des2, k=2)
#     return matches
#
#
# """
# 优化匹配结果
# RANSAC算法是RANdom SAmple Consensus的缩写,意为随机抽样一致
# """
#
#
# def RANSAC(img1, img2, kp1, kp2, matches):
#     MIN_MATCH_COUNT = 10
#     # store all the good matches as per Lowe's ratio test.
#     matchType = type(matches[0])
#     good = []
#     print(matchType)
#     if isinstance(matches[0], cv2.DMatch):
#         # 搜索使用的是match
#         good = matches
#     else:
#         # 搜索使用的是knnMatch
#         for m, n in matches:
#             if m.distance < 0.7 * n.distance:
#                 good.append(m)
#
#     if len(good) > MIN_MATCH_COUNT:
#         src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#
#         # M: 3x3 变换矩阵.
#         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         matchesMask = mask.ravel().tolist()
#
#         # h, w = img1.shape
#         # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
#         # dst = cv2.perspectiveTransform(pts, M)
#         #
#         # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
#     else:
#         print
#         "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
#         matchesMask = None
#
#     draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
#                        singlePointColor=None,
#                        matchesMask=matchesMask,  # draw only inliers
#                        flags=2)
#
#     img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
#     draw_params1 = dict(matchColor=(0, 255, 0),  # draw matches in green color
#                         singlePointColor=None,
#                         matchesMask=None,  # draw only inliers
#                         flags=2)
#
#     img33 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params1)
#
#     # cv2.imshow("before", img33)
#     # cv2.imshow("now", img3)
#     # cv2.waitKey(0)
#     plt.imshow(img33, ), plt.show()
#     plt.imshow(img3,), plt.show()
#
#
# img1 = cv2.imread("1.jpg")
# img2 = cv2.imread("2.jpg")
# kp1, des1 = SURF(img1)
# kp2, des2 = SURF(img2)
# matches = ByFlann(img1, img2, kp1, kp2, des1, des2, "SIFT")
# RANSAC(img1, img2, kp1, kp2, matches)


import time
import cv2
from matplotlib import pyplot as plt


def match_ORB():
    img1 = cv2.imread('1.jpg', 0)
    img2 = cv2.imread('2.jpg', 0)

    # 使用SURF_create特征检测器 和 BFMatcher描述符
    orb = cv2.xfeatures2d.SURF_create(float(3000))
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # matches是DMatch对象，DMatch是以列表的形式表示，每个元素代表两图能匹配得上的点。
    bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # ===========================   输出匹配的坐标  ===================================
    # kp1的索引由DMatch对象属性为queryIdx决定，kp2的索引由DMatch对象属性为trainIdx决定
    # 获取001.png的关键点位置。可以遍历matches[:20]前20个最佳的匹配点
    x, y = kp1[matches[0].queryIdx].pt
    print(x,y)
    cv2.rectangle(img1, (int(x), int(y)), (int(x) + 2, int(y) + 2), (0, 0, 255), 2)
    cv2.imshow('001', img1)
    cv2.waitKey(0)

    # 获取002.png的关键点位置
    x2, y2 = kp2[matches[0].trainIdx].pt
    print(x2,y2)
    cv2.rectangle(img2, (int(x2), int(y2)), (int(x2) + 2, int(y2) + 2), (0, 0, 255), 2)
    cv2.imshow('002', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # ==============================================================================

    # 使用plt将两个图像的第一个匹配结果显示出来
    img3 = cv2.drawMatches(img1=img1, keypoints1=kp1,
                           img2=img2, keypoints2=kp2,
                           matches1to2=matches[:20], outImg=img2,
                           flags=2)
    return img3

if __name__ == '__main__':
    start_time = time.time()
    img3 = match_ORB()
    plt.imshow(img3)
    plt.show()
    end_time = time.time()
    print("Total Spend time：", str((end_time - start_time) / 60)[0:6] + "分钟")



