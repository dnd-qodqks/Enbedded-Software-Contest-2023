import numpy as np
import cv2
import glob
import os

def draw(img, corners, imgpts):
	

	return img


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (10, 7)
frameSize = (640, 480)




# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 15
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = []  # 3D point in real-world space
imgpoints = []  # 2D points in image plane.

# directory_path = os.chdir("../../../")
# print(directory_path)

# current_path = os.getcwd()  # 현재 경로 얻기
# parent_path = os.path.abspath(os.path.join(current_path, os.pardir))  # 이전 경로 얻기
# print(current_path)

# current_path = os.getcwd()  # 현재 경로 얻기
# subfolder_name = "calli_img_2"  # 하위 폴더 이름

# new_path = os.path.join(current_path, subfolder_name)  # 새로운 경로 만들기

# print("현재 경로:", current_path)
# print("하위 폴더 경로:", new_path)

new_path = "./image"

images = glob.glob(os.path.join(new_path, "frame_*.jpg"))
print(images)
images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))
# images = glob.glob(new_path + "/*.png")
print("---------")
print(images)
for image in images:

    img = cv2.imread(image)

    
    if img is None:
        print(f"이미지를 로드할 수 없습니다: {image}")
        continue
    else:
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)
        # print(ret)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(10)

cv2.destroyAllWindows()


# print("objpoints의 길이:", len(objpoints))
# print("imgpoints의 길이:", len(imgpoints))

############## CALIBRATION #######################################################
flags = cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5  # 필요에 따라 플래그 설정
# 카메라 보정
ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints=objpoints,
    imagePoints=imgpoints,
    imageSize=frameSize,
    cameraMatrix=None,  # None으로 설정하면자동으로 카메라 행렬 계산
    distCoeffs=None,  # None으로 설정하면 자동으로 왜곡 계수 계산
    flags=flags
)

# Solve PnP
# 3D 객체의 포즈를 추정하기 위해 3D 객체의 좌표와 2D 이미지 포인트를 사용
# 여기에서는 예시로 첫 번째 이미지에 대해서만 포즈를 추정


# 포즈 결과 출력
print("Rotation Vector:")
print(rvecs)
print("tvec:", tvecs)


# print("ret:", ret)
print("cameraMatrix:", cameraMatrix)
print("dist:", dist)


