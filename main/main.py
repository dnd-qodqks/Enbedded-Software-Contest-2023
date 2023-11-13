from detect2 import Detector
from minirobot_serial import MinirobotSerial
import time

def run():
    try:
        isDebug = True
        start_flag = False
        end_flag = -1
        
        waiting_time_loading = 2
        waiting_time_head_rotation = 0.5
        
        minirobot_serial = MinirobotSerial()
        while not(start_flag):
            if isDebug: print("Wait Start Flag...")
            start_flag = minirobot_serial.RX_start_flag()
        
        if isDebug: print("!-----Start-----!")
        detector = Detector(isDebug=isDebug)
        time.sleep(waiting_time_loading) 
        
        while start_flag:
            if isDebug: 
                print("!-----Detect Start-----!")
                detector.print_result()
            
            time.sleep(waiting_time_head_rotation) 

            if detector.detect_ball():
                # theta_ball = minirobot_serial.get_angle_head()
                # if isDebug: 
                #     print("!-----Detect Ball-----!")
                #     print(f"theta_ball: {theta_ball}")
                
                if detector.detect_pole():
                    # theta_pole = minirobot_serial.get_angle_head()
                    # if isDebug: 
                    #     print("!-----Detect Pole-----!")
                    #    print(f"theta_pole: {theta_pole}")
                    
                    detector.print_result(path_save="./img", flag_save=True)
                    
                    end_flag = 1
                else:
                    # minirobot_serial.rotation_head()
                    pass
            else:
                # minirobot_serial.rotation_head()
                pass
                
            if len(detector.pixels_queue_pole) < 20 or len(detector.pixels_queue_pole) < 20:
                continue
                
            if end_flag == 1:
                break
        
        
        theta_ball = detector.get_angle(detector.dist_ball)
        
        theta_pole = detector.get_angle(detector.dist_pole)
        
        if isDebug: print("!-----Get Coordinates-----!")
        detector.get_coordinates(theta_ball, theta_pole)
        
        if isDebug: print("!-----Get Desired Point-----!")
        detector.desired_point()
        
        if isDebug: print("!-----Walk forward-----!")
        minirobot_serial.walk_forward(detector.get_distance())
        
        print(f"theta_ball: {theta_ball}")
        print(f"theta_pole: {theta_pole}")
        print(f"birdeye_view_x_ball, birdeye_view_y_ball : {detector.birdeye_view_x_ball}, {detector.birdeye_view_y_ball}")
        print(f"birdeye_view_x_pole, birdeye_view_y_pole : {detector.birdeye_view_x_pole}, {detector.birdeye_view_y_pole}")
        print(f"desired point distance: {detector.get_distance()}")
        print(f"ball distance: {detector.dist_ball}")
        print(f"pole distance: {detector.dist_pole}")
        
    except Exception as e:
        print(e)

if __name__ == "__main__":
    run()
    
            
            
        
        
        
        
