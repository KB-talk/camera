import cv2
import numpy as np
import glob

# 棋盘格的尺寸 (宽度, 高度)
chessboard_size = (9, 14)

# 棋盘格方格的实际大小 (单位: 米)
square_size = 0.010

# 对象点，例如 (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# 存储对象点和图像点的列表
objpoints = []  # 3d point in real world space
imgpoints_l = []  # 2d points in image plane for left camera
imgpoints_r = []  # 2d points in image plane for right camera

# 图像路径
left_images = glob.glob('left_photos/*.jpg')
right_images = glob.glob('right_photos/*.jpg')

if len(left_images) != len(right_images):
    raise ValueError("左右图像数量不匹配")

for left_img, right_img in zip(sorted(left_images), sorted(right_images)):
    img_l = cv2.imread(left_img)
    img_r = cv2.imread(right_img)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, None)

    if ret_l and ret_r:
        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)

        # 绘制并显示角点
        cv2.drawChessboardCorners(img_l, chessboard_size, corners_l, ret_l)
        cv2.drawChessboardCorners(img_r, chessboard_size, corners_r, ret_r)
        cv2.imshow('Left Image', img_l)
        cv2.imshow('Right Image', img_r)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 校准左相机
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, gray_l.shape[::-1], None, None)

# 校准右相机
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, gray_r.shape[::-1], None, None)


# 立体校正
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1],
    criteria=criteria, flags=flags
)

# 计算校正映射
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1], R, T)

# 输出校正参数
print('输出校正参数')
print("R1:\n", R1)
print("R2:\n", R2)
print("P1:\n", P1)
print("P2:\n", P2)
print("Q:\n", Q)
print("roi1:", roi1)
print("roi2:", roi2)

# 计算重映射矩阵
map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, gray_l.shape[::-1], cv2.CV_32FC1)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, gray_r.shape[::-1], cv2.CV_32FC1)
print('输出重映射矩阵')
print("map1_l:\n", map1_l)
print("map2_l:\n", map2_l)
print("map1_r:\n", map1_r)
print("map2_r:\n", map2_r)

# 应用校正
img_l = cv2.imread('left_photos/l0.jpg')
img_r = cv2.imread('right_photos/r0.jpg')
dst_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
dst_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

# 显示校正后的图像
cv2.imshow('Left Image', img_l)
cv2.imshow('Right Image', img_r)

cv2.imshow('Corrected Left Image', dst_l)
cv2.imshow('Corrected Right Image', dst_r)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 拼接校正后的图像
def stack_images(img1, img2):
    """水平拼接图像，方便对比"""
    height = max(img1.shape[0], img2.shape[0])
    width = img1.shape[1] + img2.shape[1]
    stacked_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 将左图和右图分别放置
    stacked_image[:img1.shape[0], :img1.shape[1], :] = img1
    stacked_image[:img2.shape[0], img1.shape[1]:, :] = img2
    return stacked_image

# 拼接校正后的图像
comparison_image = stack_images(dst_l, dst_r)

# 绘制水平线，便于对比
for i in range(0, comparison_image.shape[0], 50):  # 每隔50像素画一条线
    cv2.line(comparison_image, (0, i), (comparison_image.shape[1], i), (0, 255, 0), 1)

# 显示拼接图像
cv2.imshow('Stereo Rectification Comparison', comparison_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


