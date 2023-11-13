import cv2
import numpy as np

def nothing(x):
    pass

# 캠에서 이미지를 받아올 캡처 객체 생성
W_View_size = 800  #320  #640
H_View_size = int(W_View_size / 1.333)

cap = cv2.VideoCapture(0)

cv2.namedWindow('origin frame')
cv2.namedWindow('HSV frame')
cv2.namedWindow('YUV frame')
cv2.namedWindow('Combined frame')

margin = 100
width_track_bar = 300
height_track_bar = 100

black_image = np.zeros((int(H_View_size/2), int(W_View_size/2), 3), np.uint8)
cv2.imshow('HSV Track Bar', black_image)
cv2.imshow('YUV Track Bar', black_image)

cv2.moveWindow('origin frame', margin, margin)
cv2.moveWindow('HSV frame', width_track_bar + margin, margin)
cv2.moveWindow('YUV frame', (width_track_bar + margin)*2, margin)
cv2.moveWindow('Combined frame', (width_track_bar + margin)*3, margin)

# -----------
cv2.namedWindow('HSV Track Bar')
cv2.namedWindow('YUV Track Bar')

black_image = np.zeros((10, width_track_bar, 3), np.uint8)
cv2.imshow('HSV Track Bar', black_image)
cv2.imshow('YUV Track Bar', black_image)

cv2.createTrackbar('Max HSV H', 'HSV Track Bar', 255, 255, nothing)
cv2.createTrackbar('Min HSV H', 'HSV Track Bar', 100, 255, nothing)
cv2.createTrackbar('Max HSV S', 'HSV Track Bar', 255, 255, nothing)
cv2.createTrackbar('Min HSV S', 'HSV Track Bar', 100, 255, nothing)
cv2.createTrackbar('Max HSV V', 'HSV Track Bar', 255, 255, nothing)
cv2.createTrackbar('Min HSV V', 'HSV Track Bar', 100, 255, nothing)

cv2.createTrackbar('Max YUV Y', 'YUV Track Bar', 255, 255, nothing)
cv2.createTrackbar('Min YUV Y', 'YUV Track Bar', 100, 255, nothing)
cv2.createTrackbar('Max YUV U', 'YUV Track Bar', 255, 255, nothing)
cv2.createTrackbar('Min YUV U', 'YUV Track Bar', 100, 255, nothing)
cv2.createTrackbar('Max YUV V', 'YUV Track Bar', 255, 255, nothing)
cv2.createTrackbar('Min YUV V', 'YUV Track Bar', 100, 255, nothing)

cv2.resizeWindow(winname='HSV Track Bar', width=width_track_bar, height=height_track_bar)
cv2.resizeWindow(winname='YUV Track Bar', width=width_track_bar, height=height_track_bar)

cv2.moveWindow('HSV Track Bar', margin, H_View_size + margin)
cv2.moveWindow('YUV Track Bar', width_track_bar + margin*2, H_View_size + margin)

while True:
    # 캠에서 프레임 읽기
    _, frame = cap.read()

    # ---------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # ---------
    max_h = cv2.getTrackbarPos('Max HSV H', 'HSV Track Bar')
    min_h = cv2.getTrackbarPos('Min HSV H', 'HSV Track Bar')
    max_s = cv2.getTrackbarPos('Max HSV S', 'HSV Track Bar')
    min_s = cv2.getTrackbarPos('Min HSV S', 'HSV Track Bar')
    max_hsv_v = cv2.getTrackbarPos('Max HSV V', 'HSV Track Bar')
    min_hsv_v = cv2.getTrackbarPos('Min HSV V', 'HSV Track Bar')
    
    max_y = cv2.getTrackbarPos('Max YUV Y', 'YUV Track Bar')
    min_y = cv2.getTrackbarPos('Min YUV Y', 'YUV Track Bar')
    max_u = cv2.getTrackbarPos('Max YUV U', 'YUV Track Bar')
    min_u = cv2.getTrackbarPos('Min YUV U', 'YUV Track Bar')
    max_yuv_v = cv2.getTrackbarPos('Max YUV V', 'YUV Track Bar')
    min_yuv_v = cv2.getTrackbarPos('Min YUV V', 'YUV Track Bar')

    # ---------
    lower_hsv = np.array([min_h, min_s, min_hsv_v])
    upper_hsv = np.array([max_h, max_s, max_hsv_v])

    lower_yuv = np.array([min_y, min_u, min_yuv_v])
    upper_yuv = np.array([max_y, max_u, max_yuv_v])
    # ---------
    
    # ---------
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_yuv = cv2.inRange(yuv, lower_yuv, upper_yuv)
    mask_combined = cv2.bitwise_or(mask_hsv, mask_yuv)
    # ---------
    
    cnts_hsv = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts_yuv = cv2.findContours(mask_yuv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts_com = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    black_image_hsv = np.zeros((H_View_size, W_View_size, 3), np.uint8)
    black_image_yuv = np.zeros((H_View_size, W_View_size, 3), np.uint8)
    black_image_com = np.zeros((H_View_size, W_View_size, 3), np.uint8)
    
    cv2.drawContours(black_image_hsv, cnts_hsv, -1, (255, 255, 255), 1)
    cv2.drawContours(black_image_yuv, cnts_yuv, -1, (255, 255, 255), 1)
    cv2.drawContours(black_image_com, cnts_com, -1, (255, 255, 255), 1)

    frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    black_image_hsv = cv2.resize(black_image_hsv, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    black_image_yuv = cv2.resize(black_image_yuv, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    black_image_com = cv2.resize(black_image_com, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    
    # 이미지 출력
    cv2.imshow('origin frame', frame)
    cv2.imshow('HSV frame', black_image_hsv)
    cv2.imshow('YUV frame', black_image_yuv)
    cv2.imshow('Combined frame', black_image_com)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 객체와 윈도우 종료
cap.release()
cv2.destroyAllWindows()
