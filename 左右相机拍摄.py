
import cv2
import os

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def open_and_display_cameras():
    # 创建存储照片的文件夹
    left_folder = 'left_photos'
    right_folder = 'right_photos'
    create_directory_if_not_exists(left_folder)
    create_directory_if_not_exists(right_folder)

    # 打开两个摄像头
    left_camera = cv2.VideoCapture(0)  # 左摄像头
    right_camera = cv2.VideoCapture(1)  # 右摄像头

    if not left_camera.isOpened() or not right_camera.isOpened():
        print("无法打开摄像头")
        return

    photo_count = 0

    while True:
        # 读取帧
        ret_left, frame_left = left_camera.read()
        ret_right, frame_right = right_camera.read()

        if not ret_left or not ret_right:
            print("无法获取帧")
            break

        # 显示帧
        cv2.imshow('left_camera', frame_left)
        cv2.imshow('right_camera', frame_right)

        # 按下空格键拍照
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            left_photo_path = os.path.join(left_folder, f'l{photo_count}.jpg')
            right_photo_path = os.path.join(right_folder, f'r{photo_count}.jpg')
            cv2.imwrite(left_photo_path, frame_left)
            cv2.imwrite(right_photo_path, frame_right)
            print(f"照片已保存: {left_photo_path}, {right_photo_path}")
            photo_count += 1

        # 按下 'q' 键退出循环
        if key == ord('q'):
            break

    # 释放摄像头资源
    left_camera.release()
    right_camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_and_display_cameras()