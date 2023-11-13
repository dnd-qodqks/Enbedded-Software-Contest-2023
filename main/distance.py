import numpy as np
import cv2
import glob
import os

def smooth_image(image, kernel_size=(5, 5), sigma=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred

def find_distance(center_x, center_y, cameraMatrix, tvec, R):
    
    u = (center_x - cameraMatrix[0, 2])/ cameraMatrix[0, 0]
    v = (center_y - cameraMatrix[1, 2])/ cameraMatrix[1, 1]
    print("(u, v)", u, v)

    # R, _ = cv2.Rodrigues(rvec)
    transposed_rvec = np.transpose(R)

    cc = np.zeros((3, 1))
    # cc = np.array([[0 ,0, -325]])
    
    pc = np.vstack((u,v,1))
    print("p_c", pc)
    print(pc)
    
    tvec = tvec.T
    
    pw = transposed_rvec @ (pc - tvec)
    cw = transposed_rvec @ (cc - tvec)
    # cw[2] = 325
    print(cc.shape)
    print(tvec.shape)
    k = - cw[2] / (pw[2] - cw[2])
    
    p = cw + k * (pw - cw)
    print("cw", cw)
    print("k", k)
    print("pw", pw)
    print("p", p)
    
    p_squared_sum = np.sum(np.square(p))
    p_sum_sqrt = np.sqrt(p_squared_sum)
    p_squared_sum_cm = p_sum_sqrt / 10

    return p_squared_sum_cm

def goal_coord(r, theta):
    goal_x = r * np.cos(theta)
    goal_y = r * np.sin(theta)
    return goal_x, goal_y
