#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/ArmPi_mini/')
import cv2
import time
import math
import Camera
import yaml_handle
import json
import numpy as np
from ArmIK.ArmMoveIK import *
import HiwonderSDK.Misc as Misc
import HiwonderSDK.Board as Board
from ApexUtil import *
from HiwonderSDK.Board import setPWMServosPulse
from ApexAgent import *
from collections import deque

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

AK = ArmIK()
movetime = 3000

def initMove():
    Board.setPWMServoPulse(1, 1500, 800)
  #  AK.setPitchRangeMoving((0, 15, 5), -90, -90, 0, 1500)
    setPWMServosPulse([movetime, 4, 
                   3, 500,   
                   4, 2171,   
                   5, 1500,   
                   6, 1500])  
    time.sleep(movetime / 1000.0)

def moveto(pos):
    print(pos)
    # Reset Movement
    if pos == (0,18,0.5):
        setPWMServosPulse([movetime, 4, 
                   3, 1602,    
                   4, 1630,  
                   5, 2487,    
                   6, 1500])   
        time.sleep(movetime/1000.0)
        print("="*20)
    else:
       AK.setPitchRangeMoving(pos, -90, -90, 0, 1500)
    input()


range_rgb = {
    'red': (0, 0, 255),
    'green': (255, 0, 0)
}

lab_data = None

def load_config():
    global lab_data
    lab_data = yaml_handle.get_yaml_data(yaml_handle.lab_file_path)

def getAreaMaxContour(contours):
    contour_area_max = 0
    areaMaxContour = None
    for c in contours:
        area = abs(cv2.contourArea(c))
        if area > contour_area_max and area > 300:
            contour_area_max = area
            areaMaxContour = c
    return areaMaxContour

size = (320, 240)
__isRunning = False
snapshot_buffer = []
output_log = []

def run(img):
    global lab_data, __isRunning
    img_h, img_w = img.shape[:2]
    if not __isRunning:
        return img, None, None

    frame_resize = cv2.resize(img, size)
    frame_gb = cv2.GaussianBlur(frame_resize, (3, 3), 3)
    frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)

    results = {}
    for color in ('red', 'green'):
        if color not in lab_data:
            continue
        mask = cv2.inRange(
            frame_lab,
            tuple(lab_data[color]['min']),
            tuple(lab_data[color]['max'])
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, None)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        contour = getAreaMaxContour(contours)
        if contour is not None:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            cx = int(Misc.map(cx, 0, size[0], 0, img_w))
            cy = int(Misc.map(cy, 0, size[1], 0, img_h))
            radius = int(Misc.map(radius, 0, size[0], 0, img_w))
            results[color] = (cx, cy)
            cv2.circle(img, (cx, cy), radius, range_rgb[color], 2)
            cv2.putText(img, f'{color}: ({cx},{cy})', (cx - 40, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, range_rgb[color], 2)

    return img, results.get('red'), results.get('green')

N = 5                     
buffer_len = 2 * N           
pos_buffer = deque(maxlen=buffer_len)
time_buffer = deque(maxlen=buffer_len)


def record(pos):
    pos_buffer.append(np.array(pos))
    time_buffer.append(time.perf_counter())

def compute_velocity():
    if len(pos_buffer) < buffer_len:
        return None 
    first_pos  = list(pos_buffer)[:N]
    second_pos = list(pos_buffer)[N:]
    first_time = list(time_buffer)[:N]
    second_time= list(time_buffer)[N:]

    mean_p1 = np.mean(first_pos,  axis=0)
    mean_p2 = np.mean(second_pos, axis=0)
    t1 = np.mean(first_time)
    t2 = np.mean(second_time)

    v = (mean_p2 - mean_p1) / (t2 - t1)
    return v



pos_buf_red   = deque(maxlen=buffer_len)
time_buf_red  = deque(maxlen=buffer_len)
pos_buf_green = deque(maxlen=buffer_len)
time_buf_green= deque(maxlen=buffer_len)


def process_raw_pos(pos):
    if pos  == None:
        return None
    return ((pos[0]-445)/(410/18),pos[1]/(410/18))
if __name__ == '__main__':

    initMove()
    load_config()
    __isRunning = True
    cap = cv2.VideoCapture('http://127.0.0.1:8080?action=stream')

    frame_count = 1
    while True:
        ret, img = cap.read()
        if ret:
            frame = img.copy()
            frame, red_pos, green_pos = run(frame)
            red_pos = process_raw_pos(red_pos)
            green_pos = process_raw_pos(green_pos)
            
            now = time.perf_counter()
            dt = 10000
            if len(time_buf_red) == buffer_len:
                t_third  = time_buf_red[2]  # frame 3
                t_eighth = time_buf_red[7]  # frame 8

                dt = t_eighth - t_third

                print(f"dt between frame 3 and 8: {dt:.6f} s")
            
            predicted_danger = False
            
            if red_pos:
                pos_buf_red.append(red_pos)
                time_buf_red.append(now)
                if len(pos_buf_red) == buffer_len:
                    v_red = (np.mean(list(pos_buf_red)[N:],axis=0)
                           - np.mean(list(pos_buf_red)[:N], axis=0)) \
                           / (np.mean(list(time_buf_red)[N:]) 
                              - np.mean(list(time_buf_red)[:N]))
                else:
                    v_red = None

            if green_pos:
                pos_buf_green.append(green_pos)
                time_buf_green.append(now)
                if len(pos_buf_green) == buffer_len:
                    v_green = (np.mean(list(pos_buf_green)[N:],axis=0)
                             - np.mean(list(pos_buf_green)[:N], axis=0)) \
                             / (np.mean(list(time_buf_green)[N:]) 
                                - np.mean(list(time_buf_green)[:N]))
                    print(v_green)

                    predicted_danger = predict_collision(list(pos_buf_red), list(pos_buf_green),N, dt)
                else:
                    v_green = None            

            agent = LLM_Agent()
            
            output_log.append({
                "frame_time": now,
                "red_pos": red_pos,
                "green_pos": green_pos,
                "danger": predicted_danger
            })
            if predicted_danger:
                snapshot = build_snapshot(list(pos_buf_red),list(pos_buf_green),v_red,v_green,N)
                start_time = time.perf_counter()
                t_pt = agent.decide_move_kid_apex(snapshot,select_safe_targets(snapshot)[0]['point'])
                elapsed = time.perf_counter() - start_time
                print('-'*20)
                print("total decision time:"+str(elapsed))
                moveto(t_pt)
            
            # Comparsion Expt
           # if green_pos is not None and red_pos is not None:
            #    print(v_red,v_green,frame_count)
             #   if frame_count % 100 == 0:
              #       start_time = time.perf_counter()
               #      snapshot = build_snapshot(list(pos_buf_red),list(pos_buf_green),v_red,v_green,N)
                #     t_pt = agent.decide_move_box(snapshot,get_points())
                 #    elapsed = time.perf_counter() - start_time
                  #   print("total decision time:"+str(elapsed))
                   #  moveto(t_pt)
                     
            print(f"Red: {red_pos}, green: {green_pos}, Danger: {predicted_danger}")
            frame_resize = cv2.resize(frame, size)
            cv2.imshow('frame', frame_resize)
            frame_count += 1 
            if cv2.waitKey(1) == 27:
                break
            time.sleep(0.01)
        else:
            time.sleep(0.01)

    with open("output.json", "w") as f:
        json.dump(output_log, f, indent=2)
    cv2.destroyAllWindows()
