import cv2
import numpy as np
import glob

# 棋盘格参数
chessboard_size = (9, 14)  # 棋盘格内角点的数量
square_size = 1.0  # 棋盘格方块的边长

# 准备对象点，例如 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# 存储对象点和图像点
objpoints = []  # 3D点在世界坐标系中的位置
imgpoints = []  # 2D点在图像平面的位置

# 图像路径
images = glob.glob('left_photos/*.jpg')

if not images:
    raise FileNotFoundError("没有找到图像文件，请检查路径是否正确")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"无法读取图像: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 绘制和显示角点
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    else:
        print(f"未能在图像 {fname} 中找到棋盘格角点")

cv2.destroyAllWindows()

# 检查 objpoints 和 imgpoints 是否为空
if len(objpoints) == 0 or len(imgpoints) == 0:
    raise ValueError("没有找到足够的棋盘格角点，无法进行相机标定")

# 打印调试信息
print(f"找到 {len(objpoints)} 张图像的角点")

# 标定相机
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n", mtx)
print("Distortion coefficients : \n", dist)
print("Rotation vectors : \n", rvecs)
print("Translation vectors : \n", tvecs)
