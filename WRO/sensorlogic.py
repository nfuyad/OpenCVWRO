import threading
import serial

ultra_data= [-1.0, -1.0, -1.0]
roat= None

def read_serial():
    global ultra_data
    global roat
    ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            # Expected format: F:45 L:100 R:98
            line= line.split(",")
            ultra_data[0]= float(line[0])
            ultra_data[1]= float(line[1])
            ultra_data[2]= float(line[2])

            roat= float(line[3])

            print(ultra_data)
            

        except Exception as e:
            print("Serial error:", e)


read_serial()
