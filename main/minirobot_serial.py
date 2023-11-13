import serial
import time

class MinirobotSerial():
    def __init__(self) -> None:
        
        BPS =  4800  # 4800,9600,14400, 19200,28800, 57600, 115200
        self.serial_port = serial.Serial('/dev/ttyS0', BPS, timeout=0.01)
        self.serial_port.flush() # serial cls
        
        self.flag_get_head_angle = 37
        self.left_rotation_head_data = 33
        self.right_rotation_head_data = 34
        self.init_rotation_head_data = 23
        self.up_rotation_head_data = 35
        self.down_rotation_head_data = 36
        self.rotation_body_data = {"left_turn_5": 1, "left_turn_10": 4, "left_turn_20": 8, "left_turn_45": 22, "left_turn_60": 25,
                                   "right_turn_5": 3, "right_turn_10": 6, "right_turn_20": 9, "right_turn_45": 24, "right_turn_60": 19}
        
    #-----------------------------------------------
    def TX_data(self, one_byte) -> None:  # one_byte= 0~255

        #ser.write(chr(int(one_byte)))          #python2.7
        self.serial_port.write(serial.to_bytes([one_byte]))  #python3

    #-----------------------------------------------
    def RX_data(self) -> int:

        if self.serial_port.inWaiting() > 0:
            result = self.serial_port.read(1)
            RX = ord(result)
            return RX
        else:
            return 0

    #-----------------------------------------------
    def check_flag(self, tx_data, rx_data):
        
        while True:
            data = 0
            self.TX_data(tx_data)
            
            data = self.RX_data()
            if data == rx_data:
                break
            
            print(f"Check Data: tx_data({tx_data}), rx_data({rx_data})...")

        print("!-----Checked Data-----!")
    
    #-----------------------------------------------
    def RX_start_flag(self) -> bool:
        
        try:
            data = self.RX_data()
            
            print(f"data: {data}")
            
            if data == 5:
                return True
            else:
                return False
                    
        except Exception as e:
            print(e)
            return False

    #-----------------------------------------------
    def get_angle_head(self) -> int:
        
        try:
            while True:
                data = 0
                self.TX_data(self.flag_get_head_angle) # 37
                
                data = self.RX_data()
                if data == 195:
                    break
                
                print(f"tx_data: {self.flag_get_head_angle}, data: {data}")
            
            while True:
                angle = 0
                self.TX_data(200)
                time.sleep(1e-3)
                angle = self.RX_data()
                if angle != 0 and angle != 195 and angle != 230:
                    print("break angle")
                    break
                
                print(f"tx_data: {200}, angle: {angle}")
            print(f"!!!!!!!!!!!!!!!angle: {angle}!!!!!!!!!!!!!!")    
            
            # self.check_flag(self.flag_get_head_angle, 380)
            
            # self.check_flag(777, 380)
            
            return angle
            
        except Exception as e:
            print(e)
            return 0

    #-----------------------------------------------
    def rotation_head(self) -> None:
        
        try:
            angle_head = self.get_angle_head()
            
            # if angle_head < 20:
            #     self.check_flag(self.right_rotation_head_data, 230)
            # elif angle_head > 180:
            #     self.check_flag(self.left_rotation_head_data, 230)
            # else:
            #     if self.flag_head == 0:
            #         self.check_flag(self.right_rotation_head_data, 230)
            #     elif self.flag_head == 1:
            #         self.check_flag(self.left_rotation_head_data, 230)
                
            if angle_head > 170 and angle_head != 195 and angle_head != 230:
                self.check_flag(self.init_rotation_head_data, 230)
                print("###### Init rotation head ######")
                time.sleep(3)
            else:
                self.check_flag(self.right_rotation_head_data, 230)
                time.sleep(1.5)
                
        except Exception as e:
            print(e)

    #-----------------------------------------------
    def walk_forward(self, distance):
        try:
            call_number = int(distance / 4)
            for _ in range(call_number):
                self.check_flag(31, 230)
                time.sleep(3)
                        
            self.check_flag(26, 230)
        except Exception as e:
            print(e)
        
    #-----------------------------------------------
    
