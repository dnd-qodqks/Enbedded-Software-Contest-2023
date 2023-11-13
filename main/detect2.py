# 상대 오차 거리 적용 https://github.com/TareDevarsh/distance_fromcamera

import os
import cv2
import numpy as np
import math
from distance import *
from collections import deque
        
class Detector():
    def __init__(self, isDebug=False) -> None:
        self.isDebug = isDebug
        self.gain_desired_point = 1
        
        ################## CV2 ####################
        self.cap = cv2.VideoCapture(0)
        
        if self.cap.isOpened(): 
            print('The camera is successfully connected.')
        else:
            print('Failed to connect to the camera.')
        
        self.W_View_size = 640 #320  #640
        self.H_View_size = int(self.W_View_size / 1.333)
        
        _, frame = self.cap.read()
        
        if self.isDebug == True:
            cv2.namedWindow("cam")
            cv2.namedWindow("result")
            cv2.namedWindow("mask")
            
            cv2.moveWindow("cam", 0, 0)
            cv2.moveWindow("result", self.W_View_size+15, 35)
            
            black_image = np.zeros((self.H_View_size, self.W_View_size, 3), np.uint8)
            
            cv2.imshow("cam", frame)
            cv2.imshow("result", black_image)
            cv2.imshow("mask", black_image)
            cv2.waitKey(1500)
        ###########################################
        
        self.hsv_ball_lower = (130, 100, 150)
        self.hsv_ball_upper = (210, 190, 255)
        self.yuv_ball_lower = (65, 100, 145)
        self.yuv_ball_upper = (200, 215, 255)
        
        self.hsv_pole_lower = (25, 40, 170)
        self.hsv_pole_upper = (60, 160, 255)
        self.yuv_pole_lower = (190, 60, 55)
        self.yuv_pole_upper = (220, 115, 145)
        
        ############## object point ###############
        self.x4_ball = -1
        self.y4_ball = -1
        self.w4_ball = -1
        self.h4_ball = -1
        self.center_x_ball = -1
        self.center_y_ball = -1
        self.distance_center2ball = -1
        self.theta_center2ball = -1
        self.isBall = False
        self.birdeye_view_x_ball = -1
        self.birdeye_view_y_ball = -1
        
        # bird-eye
        self.p1 = [450, 450] # left up
        self.p2 = [800, 450] # left low
        self.p3 = [1000, 700] # right up
        self.p4 = [250, 700] # left up
        
        self.x4_pole = -1
        self.y4_pole = -1
        self.w4_pole = -1
        self.h4_pole = -1
        self.center_x_pole = -1
        self.center_y_pole = -1
        self.distance_center2pole = -1
        self.theta_center2pole = -1
        self.isPole = False
        self.birdeye_view_x_pole = -1
        self.birdeye_view_y_pole = -1
        
        self.pixels_queue_ball = deque(maxlen = 20)
        self.pixels_queue_pole = deque(maxlen = 20)

        ###########################################
        
        ######## Camera Calllibration #############
        self.chessboardSize = (11,8)
        self.frameSize = (640, 480)
        self.cameraMatrix = np.array([
            [515.17409226, 0., 305.51382975],
            [0., 514.82512516, 227.44052074],
            [0., 0., 1.]
        ])

        self.rvec = np.zeros((3, 1))

        self.tvec = np.array([[-79.98565484 + 315], [141.22660003], [374.21090245]])
        ###########################################
    
    #-----------------------------------------------
    def detect_ball(self) -> bool:
        
        try:
            _, frame = self.cap.read()
            if self.isDebug == True:
                cv2.imwrite("./ball.jpg", frame)
                cv2.imshow("cam", frame)
                cv2.waitKey(1)
                
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            hsv_Lower = np.array(self.hsv_ball_lower)
            hsv_Upper = np.array(self.hsv_ball_upper)
            yuv_Lower = np.array(self.hsv_ball_lower)
            yuv_Upper = np.array(self.hsv_ball_upper)
            #mask = cv2.inRange(hsv, hsv_Lower, hsv_Upper)
            
            hsv_mask = cv2.inRange(hsv, hsv_Lower, hsv_Upper)
            # YUV에서 공의 마스크 생성
            yuv_mask = cv2.inRange(yuv, yuv_Lower, yuv_Upper)
            # HSV와 YUV의 교집합 생성
            #intersection = cv2.bitwise_and(hsv_mask, yuv_mask)
            # 교집합 마스크를 사용하여 원본 이미지에서 공을 추적
            #mask = cv2.bitwise_and(frame, frame, mask=intersection)
            mask = cv2.bitwise_or(hsv_mask, yuv_mask)
            
            
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            
            black_image = np.zeros((self.H_View_size, self.W_View_size, 3), np.uint8)
            
            cv2.drawContours(black_image, cnts, -1, (255, 255, 255), 2)
            cv2.imshow("mask", black_image)
            
            cv2.waitKey(1)
            
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)

                Area = cv2.contourArea(c) / 50
                if Area > 255:
                    Area = 255

                if Area <= 255:
                    self.rect_ball = cv2.minAreaRect(c)
                    print(f"rect_ball: {self.rect_ball}")
                    x4, y4, w4, h4 = cv2.boundingRect(c)
                    self.x4_ball = x4
                    self.y4_ball = y4
                    self.w4_ball = w4
                    self.h4_ball = h4
                    
                    self.center_x_ball = int(self.x4_ball + self.w4_ball / 2)
                    self.center_y_ball = int(self.y4_ball + self.h4_ball / 2)

                    self.distance_center2ball = int(self.W_View_size/2) - self.center_x_ball
                    
                    self.isBall = True
                    
                    return True
                else:
                    self.x4_ball = -1
                    self.y4_ball = -1
                    self.w4_ball = -1
                    self.h4_ball = -1
                    
                    self.center_x_ball = -1
                    self.center_y_ball = -1
                    
                    self.isBall = False
                
                return False
            
            else:
                return False
            
        except Exception as e:
            print(e)
            return False

    #-----------------------------------------------    
    def detect_pole(self) -> bool:
        
        try:
            _, frame = self.cap.read()
            if self.isDebug == True:
                cv2.imshow("cam", frame)
                cv2.waitKey(1)
                
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_Lower = self.hsv_pole_lower
            hsv_Upper = self.hsv_pole_upper
            
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            yuv_Lower = self.yuv_pole_lower
            yuv_Upper = self.yuv_pole_upper
            
            hsv_mask = cv2.inRange(hsv, hsv_Lower, hsv_Upper)
            yuv_mask = cv2.inRange(yuv, yuv_Lower, yuv_Upper)
            
            mask = cv2.bitwise_and(hsv_mask, yuv_mask)
            
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)

                Area = cv2.contourArea(c) / 50
                if Area > 255:
                    Area = 255
                    
                if Area <= 255:
                    x4, y4, w4, h4 = cv2.boundingRect(c)
                    self.rect_pole = cv2.minAreaRect(c)
                    print(f"rect_pole: {self.rect_pole}")
                    
                    self.x4_pole = x4
                    self.y4_pole = y4
                    self.w4_pole = w4
                    self.h4_pole = h4
                    
                    self.center_x_pole = int(self.x4_pole + self.w4_pole / 2)
                    self.center_y_pole = int(self.y4_pole + self.h4_pole / 2)

                    self.distance_center2pole = int(self.W_View_size/2) - self.center_x_pole
                    
                    self.isPole = True
                    
                    return True
                else:
                    self.x4_pole = -1
                    self.y4_pole = -1
                    self.w4_pole = -1
                    self.h4_pole = -1
                    
                    self.center_x_pole = -1
                    self.center_y_pole = -1
                    
                    self.isPole = False
                    
                return False
            
            else:
                return False
            
        except Exception as e:
            print(e)
            return False
    
    #-----------------------------------------------
    def print_result(self, path_save="./img", flag_save=False):
        if self.isDebug == True:
            _, frame = self.cap.read()
            cv2.imshow("cam", frame)
            
            if self.isBall:
                cv2.rectangle(frame, 
                            (self.x4_ball, self.y4_ball), 
                            (self.x4_ball + self.w4_ball, self.y4_ball + self.h4_ball), 
                            (0, 255, 0), 
                            2)
                
                width = 4.5
                focal = 330
                #find no of pixels covered
                pixels = self.rect_ball[1][0]
                print(f"pixels: {pixels}")
                
                # pixels를 큐에 추가
                self.pixels_queue_ball.append(pixels)
                
                # 큐에 들어있는 데이터를 평균하여 최종 pixels로 사용
                if len(self.pixels_queue_ball) > 0:
                    average_pixels = sum(self.pixels_queue_ball) / len(self.pixels_queue_ball)
                
                    print(f"final_pixels: {average_pixels}")

                    #calculate distance
                    self.dist_ball = (width*focal)/average_pixels
                    print(f"ball distance: {self.dist_ball}")
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX 
                    org = (0,20)  
                    fontScale = 0.6 
                    color = (0, 0, 255)

                    frame = cv2.putText(frame, 'Distance from Camera in CM :', org, font, 1, color, 2, cv2.LINE_AA)
                    frame = cv2.putText(frame, str(self.dist_ball), (110,50), font, fontScale, color, 1, cv2.LINE_AA)
            
            if self.isPole:
                cv2.rectangle(frame, 
                            (self.x4_pole, self.y4_pole), 
                            (self.x4_pole + self.w4_pole, self.y4_pole + self.h4_pole), 
                            (0, 0, 255), 
                            2)
                            
                width = 7.5
                focal = 330
                #find no of pixels covered
                pixels = self.rect_pole[1][0]
                
                self.pixels_queue_pole.append(pixels)
                
                # 큐에 들어있는 데이터를 평균하여 최종 pixels로 사용
                if len(self.pixels_queue_pole) > 0:
                    average_pixels = sum(self.pixels_queue_pole) / len(self.pixels_queue_pole)
                
                self.dist_pole = (width*focal)/average_pixels
                
                print(f"pole distance: {self.dist_pole}")

            
               
            cv2.imshow("result", frame)
            cv2.waitKey(1)

        if flag_save == True:
            if not os.path.exists(path_save):
                os.makedirs(path_save)
                print(f"The directory {path_save} has been created.")
            else:
                print(f"The directory {path_save} already exists.")
                
            cv2.imwrite(path_save + "/result.jpg", frame)
            
    #-----------------------------------------------
    def get_remove_noise_mask(self, mask, morphology="open", kernel_size=(3,3)):
        try:
            if morphology == "erode":
                # 구조화 요소 커널, 사각형 (3x3) 생성 ---①
                k = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
                # 침식 연산 적용 ---②
                remove_noise_mask = cv2.erode(mask, k)
                
            elif morphology == "dilate":
                # 구조화 요소 커널, 사각형 (3x3) 생성 ---①
                k = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
                # 팽창 연산 적용 ---②
                remove_noise_mask = cv2.dilate(mask, k)
                
            elif morphology == "open":
                kernel = np.ones(kernel_size, np.uint8)
                remove_noise_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
            elif morphology == "close":
                kernel = np.ones(kernel_size, np.uint8)
                remove_noise_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
            return remove_noise_mask
        
        except Exception as e:
            print(e)
            return None
       
    #-----------------------------------------------
    def window_center_point(self) -> tuple[int, int]:
        return int(self.W_View_size/2), int(self.H_View_size/2)
    
    #-----------------------------------------------
    def get_coordinates(self, theta_ball, theta_pole):
        try:
            # distance_ball = find_distance(self.center_x_ball, self.center_y_ball, self.cameraMatrix, self.tvec, self.rvec)
            # distance_pole = find_distance(self.center_x_pole, self.center_y_pole, self.cameraMatrix, self.tvec, self.rvec)
            # print(f"distance_ball: {distance_ball}")
            # print(f"distance_pole: {distance_pole}")
            
            self.birdeye_view_x_ball, self.birdeye_view_y_ball = goal_coord(self.dist_ball, theta_ball-100)
            self.birdeye_view_x_pole, self.birdeye_view_y_pole = goal_coord(self.dist_pole, theta_pole-100)
            print(f"birdeye_view_x_ball, birdeye_view_y_ball: ({self.birdeye_view_x_ball}, {self.birdeye_view_y_ball})")
            print(f"birdeye_view_x_pole, birdeye_view_y_pole: ({self.birdeye_view_x_pole}, {self.birdeye_view_y_pole})")
            
        except Exception as e:
            print(e)
        
    #-----------------------------------------------
    def desired_point(self):
        try:
            L = 15 * self.gain_desired_point #직선의 방정식(Pole과 Ball)과의 수직 거리 반경

            Ball_x, Ball_y = self.birdeye_view_x_ball, self.birdeye_view_y_ball
            Pole_x, Pole_y = self.birdeye_view_x_pole, self.birdeye_view_y_pole
            
            #Pole과 Ball의 X좌표 크기 비교
            if Ball_x != Pole_x:
                Py_By = Pole_y - Ball_y
                Px_Bx = Pole_x - Ball_x

                Desired_x1 = L*(np.abs(Py_By)/np.sqrt(np.power(Py_By, 2)+np.power(Px_Bx, 2)))+Ball_x
                Desired_y1 = L*((Px_Bx)/(Ball_y - Pole_y))*(np.abs(Py_By)/np.sqrt(np.power(Py_By, 2)+np.power(Px_Bx, 2)))+Ball_y
                
                Desired_x2 = (-1)*L*(np.abs(Py_By)/np.sqrt(np.power(Py_By, 2)+np.power(Px_Bx, 2)))+Ball_x
                Desired_y2 = (-1)*L*((Px_Bx)/(Ball_y - Pole_y))*(np.abs(Py_By)/np.sqrt(np.power(Py_By, 2)+np.power(Px_Bx, 2)))+Ball_y

                if Ball_x < Desired_x1:
                    self.Desired_x = Desired_x1
                    self.Desired_y = Desired_y1
                else:
                    self.Desired_x = Desired_x2
                    self.Desired_y = Desired_y2    

            else:
                self.Desired_x = Ball_x - L
                self.Desired_y = Ball_y

            if self.isDebug == True:
                print(f"Desired_x: {self.Desired_x}, Desired_y: {self.Desired_y}")
                
        except Exception as e:
            print(e)

    #-----------------------------------------------  
    def get_angle(self, distance):
        try:
            p1 =  [171, 1]  # 좌상
            p2 =  [0, 480] # 좌하
            p3 =  [455, 5] # 우상
            p4 = [640, 480]  # 우하
            corner_points_arr = np.float32([p1, p2, p3, p4])
            
            image_p1 = [0, 0]
            image_p2 = [0, self.H_View_size]
            image_p3 = [self.W_View_size, 0]
            image_p4 = [self.W_View_size, self.H_View_size]
            image_params = np.float32([image_p1, image_p2, image_p3, image_p4])

            mat = cv2.getPerspectiveTransform(corner_points_arr, image_params)
            # mat = 변환행렬(3*3 행렬) 반

            center_x = self.x4_ball + (self.w4_ball // 2)
            center_y = self.y4_ball + (self.h4_ball // 2)

            original_point = np.array([[center_x], [center_y], [1]])

            # 변환 행렬 적용
            transformed_point = np.dot(mat, original_point)

            # 변환된 좌표에서 x와 y 값을 추출
            transformed_x = int(transformed_point[0, 0] / transformed_point[2, 0])
            transformed_y = int(transformed_point[1, 0] / transformed_point[2, 0])
            
            l1 = 20.4 + ((480-transformed_y)/480*200)
            return math.acos(l1/distance)
            
        except Exception as e:
            print(e)

    #-----------------------------------------------
    def get_distance(self):
        distance = math.sqrt(self.Desired_x**2 +  self.Desired_y** 2)
        return distance
    
    #-----------------------------------------------  
    def get_dist(self, image):
        
        return image
    
    #-----------------------------------------------
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
        
    #-----------------------------------------------
