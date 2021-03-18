#should output a dictionary with the same format as the JSON received in RemoteControlReceiver
from src.utils.templates.workerProcess import WorkerProcess
from threading import Thread
import math
import time
import numpy as np

class AutonomousController(WorkerProcess):
    #======================= INIT =======================
    def __init__(self, inPs, outPs):
        """Accepts the detected items from the visualization
        module, and instructs nucleo
        ------------
        inPs : list(Pipe)
            0 - lanes
            1 - objects
        outPs : list(Pipe)
            0 - high-level instructions
        """
        self.img_shape = (480, 640)
        super(AutonomousController,self).__init__(inPs, outPs)

        self.SPEED_LEVELS = { 1 : 0.12,
                              2 : 0.15}
        self.current_speed_level = 1
        self.current_steer_angle = 0.0
    
    # fetches the next speed level to feed to the nucleo,
    # and updates the speed to the next level if "update"
    # is true
    def getSpeed(self,update=True):
        ret = self.SPEED_LEVELS[self.current_speed_level]
        if update:
            if self.current_speed_level == max(self.SPEED_LEVELS.keys()):
                self.current_speed_level = 1
            else:
                self.current_speed_level+=1
        
        return ret
    #======================= BASIC MOVEMENT =======================
    def drive(self,t,speed=0.2):
        #every tenth of a second (approx), we send a command to the Nucleo
        #indicating the speed we want to keep
        # "t" should be the time in seconds
        command = {
            'action' : 'MCTL',
            'speed'  : speed,
            'steerAngle' : self.current_steer_angle
        }
        for i in range(0,10*t):
            self.outPs[0].send(command)
            time.sleep(0.1)

    def bump(self,speed=0.25):
        command = {
            'action' : 'MCTL',
            'speed'  : speed,
            'steerAngle' : self.current_steer_angle
        }
        self.outPs[0].send(command)
    
    def brake(self,speed=0.0):
        command = {
            'action' : 'MCTL',
            'speed'  : speed,
            'steerAngle' : self.current_steer_angle
        }
        self.outPs[0].send(command)

    #======================= RUN =======================
    def run(self):
        """Apply the initializing methods and start the threads.
        """
        super(AutonomousController,self).run()

        #======================= RUN =======================
    def stop(self):
        """Stop the car, and then shut down the thread
        """
        command = {
            'action' : 'MCTL',
            'speed'  : 0.0,
            'steerAngle' : 0.0
        }
        self.outPs[0].send(command)

        super(AutonomousController,self).stop()

    #======================= INIT THREADS =======================
    def _init_threads(self):
        """Initialize the read thread to receive the video.
        """
        if self._blocker.is_set():
            return

        command = {
            'action' : 'MCTL',
            'speed'  : 0.0,
            'steerAngle' : 0.0
        }
        self.outPs[0].send(command)

        thr = Thread(name='ControlThread',target = self._control_thread, args= (self.inPs, self.outPs, ))
        thr.daemon = True
        self.threads.append(thr)

    def _control_thread(self, inPs, outPs):
        """Read the image from input stream, process it and send lane information

        Parameters
        ----------
        inPs : list(Pipe)
            0 - lanes
            1 - objects
        outPs : list(Pipe)
            0 - high-level instructions
        """
        #time.sleep(5)
        #self.bump()

        while True:
            try:
                self.brake(speed=0.0)
                #information from the lane detector
                stamp, detected_lane_info = inPs[0].recv()
                ld_left = detected_lane_info[0]
                ld_right = detected_lane_info[1]
                ld_intersection = detected_lane_info[2]

                #information from the object detector
                stamp, detected_obj_info = inPs[1].recv()
                if len(detected_lane_info)>0:
                    ts_list = self.check_traffic_signs(detected_obj_info)

                if ld_intersection >= 250:
                    self.routine_intersection(ld_intersection)
                else:  
                    self.routine_cruise(ld_left,ld_right)

            except Exception as e:
                print("AutonomousController failed to obtain objects:",e,"\n")
                pass

    #======================= LANE KEEPING =======================
    def routine_cruise(self,left_lane_pts, right_lane_pts):
        steering_angle = self.calculate_steering_angle(left_lane_pts,right_lane_pts)
        steering_angle = np.clip(steering_angle, -21, 21)
        new_steer_angle = float(steering_angle)
        sa_diff = self.current_steer_angle - new_steer_angle
        self.current_steer_angle = new_steer_angle
        self.bump()
        time.sleep(0.1)
         
    def calculate_steering_angle(self,left_lane_pts, right_lane_pts):
        #print("received lane array",lanes) 
        #print("~~~calculating steering angle~~~")
        height, width = self.img_shape
        x_offset = 0.0

        left_x1, left_y1, left_x2, left_y2 = left_lane_pts
        right_x1, right_y1, right_x2, right_y2 = right_lane_pts

        left_found = False if (left_x1==0 and left_y1==0 and left_x2==0 and left_y2==0) else True
        #if left_found: print("found left lane")
        right_found = False if (right_x1==0 and right_y1==0 and right_x2==0 and right_y2==0) else True
        #if right_found: print("found right lane")

        if left_found and right_found: #both lanes
            cam_mid_offset_percent = 0.02
            mid = int(width/2 * (1 + cam_mid_offset_percent))
            x_offset = (left_x2 + right_x2) / 2 - mid
        elif left_found and not right_found: #left lane only
            x_offset = left_x2 - left_x1
        elif not left_found and right_found: #right lane ony
            x_offset = right_x2 - right_x1
        else: #no lanes detected
            x_offset = 0
        
        y_offset = int(height/2)

        steering_angle = math.atan(x_offset / y_offset) #in radians
        steering_angle = int(steering_angle * 180.0 / math.pi)
        return steering_angle

    #======================= INTERSECTION HANDLING =======================
    def routine_intersection(self, intersection_y):
        print("routine_intersection: STOPPING AT INTERSECTION AT Y=%d",intersection_y)
        self.brake()
        time.sleep(3)

        # TODO: how to determine these?
        turn_left = False
        turn_right = True
        
        if turn_right:
            print("routine_intersection: making right turn")
            self.current_steer_angle = 0.0
            self.drive(t=2,speed=0.3)
            self.brake()

        elif turn_left:
            pass
        
        time.sleep(10)

    #======================= TRAFFIC SIGN HANDLING =======================
    def check_traffic_signs(self,obj_info):
        for o in obj_info:
            print(o)

        



