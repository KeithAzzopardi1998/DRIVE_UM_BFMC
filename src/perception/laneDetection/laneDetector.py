import numpy as np
import cv2
import time
import math
from threading import Thread
from collections import deque
from scipy import stats
import src.perception.laneDetection.preProcessing as pp

from src.utils.templates.workerProcess import WorkerProcess

class LaneDetector(WorkerProcess):
    #======================= INIT =======================
    def __init__(self, inPs, outPs):
        """Accepts frames from the camera, processes the frame and detect lanes,
        and transmits information about left and right lanes.
        Parameters
        ------------
        inPs : list(Pipe)
            0 - receive image feed from the camera
        outPs : list(Pipe)
            0 - send lane information
        """

        self.img_shape = (480, 640)
        height = self.img_shape[0]
        width = self.img_shape[1]

        region_top_left = (0.15*width, 0.4*height)
        region_top_right = (0.85*width, 0.4*height)
        region_bottom_left_A = (0.00*width, 1.00*height)
        region_bottom_left_B = (0.00*width, 0.8*height)
        region_bottom_right_A = (1.00*width, 1.00*height)
        region_bottom_right_B = (1.00*width, 0.8*height)

        self.mask_vertices = np.array([[region_bottom_left_A,
                                         region_bottom_left_B,
                                         region_top_left,
                                         region_top_right,
                                         region_bottom_right_B,
                                         region_bottom_right_A]], dtype=np.int32)


        super(LaneDetector,self).__init__(inPs, outPs)

    #======================= RUN =======================
    def run(self):
        """Apply the initializing methods and start the threads.
        """
        super(LaneDetector,self).run()

    #======================= INIT THREADS =======================
    def _init_threads(self):
        """Initialize the read thread to receive the video.
        """
        if self._blocker.is_set():
            return

        thr = Thread(name='StreamSending',target = self._the_thread, args= (self.inPs, self.outPs, ))
        thr.daemon = True
        self.threads.append(thr)


    #======================= METHODS =======================
    def laneDetection(self, img_in):
        try:
            # Setting Hough Transform Parameters
            rho = 1 # 1 degree
            theta = (np.pi/180) * 1
            threshold = 15
            min_line_length = 20
            max_line_gap = 10

            left_lane_coefficients  = pp.create_coefficients_list()
            right_lane_coefficients = pp.create_coefficients_list()

            previous_left_lane_coefficients = None
            previous_right_lane_coefficients = None


            # Begin lane detection pipiline
            img = img_in.copy()
            img = cv2.convertScaleAbs(img,alpha=2.0,beta=30)
            #print("LANE LINES LOG - COPIED IMAGE", img)
            combined_hsl_img = pp.filter_img_hsl(img)
            #print("LANE LINES LOG - COMBINED IMAGE HSL", combined_hsl_img)
            grayscale_img = pp.grayscale(combined_hsl_img)
            #print("LANE LINES LOG - COMBINED IMAGE GRAYSCALE", grayscale_img)
            gaussian_smoothed_img = pp.gaussian_blur(grayscale_img, kernel_size=5)
            canny_img = cv2.Canny(gaussian_smoothed_img, 50, 150)
            segmented_img = pp.getROI(canny_img,self.mask_vertices)
            #print("LANE LINES LOG - SEGMENTED IMAGE SUM", np.sum(segmented_img))
            hough_lines = pp.hough_transform(segmented_img, rho, theta, threshold, min_line_length, max_line_gap)
            #print("LANE LINES LOG - HOUGH LINES", hough_lines)

            preprocessed_img = cv2.cvtColor(segmented_img,cv2.COLOR_GRAY2BGR)
            left_lane_lines, right_lane_lines = pp.separate_lines(hough_lines, img)
        except Exception as e:
            print("lane preprocessing failed")
            #return np.array([[0.0,0.0], [0.0,0.0]], img_in
            left_lane_lines = []
            right_lane_lines = []
            preprocessed_img = img_in

        #print("test")
        try:
            left_lane_slope, left_intercept = pp.getLanesFormula(left_lane_lines)        
            smoothed_left_lane_coefficients = pp.determine_line_coefficients(left_lane_coefficients, [left_lane_slope, left_intercept])
        except Exception as e:
            print("Using saved coefficients for left coefficients", e)
            smoothed_left_lane_coefficients = pp.determine_line_coefficients(left_lane_coefficients, [0.0, 0.0])
            
        try: 
            right_lane_slope, right_intercept = pp.getLanesFormula(right_lane_lines)
            smoothed_right_lane_coefficients = pp.determine_line_coefficients(right_lane_coefficients, [right_lane_slope, right_intercept])
        except Exception as e:
            print("Using saved coefficients for right coefficients", e)
            smoothed_right_lane_coefficients = pp.determine_line_coefficients(right_lane_coefficients, [0.0, 0.0])

        return np.array([smoothed_left_lane_coefficients, smoothed_right_lane_coefficients]), preprocessed_img

        
    def points_from_lane_coeffs(self,line_coefficients):
        A = line_coefficients[0]
        b = line_coefficients[1]

        if A==0.00 and b==0.00:
            return [0,0,0,0]

        height, width = self.img_shape

        bottom_y = height - 1
        top_y = self.mask_vertices[0][1][1]
        # y = Ax + b, therefore x = (y - b) / A
        bottom_x = (bottom_y - b) / A
        # clipping the x values
        bottom_x = min(bottom_x, 2*width)
        bottom_x = max(bottom_x, -1*width)

        top_x = (top_y - b) / A
        # clipping the x values
        top_x = min(top_x, 2*width)
        top_x = max(top_x, -1*width)

        return [int(bottom_x), int(bottom_y), int(top_x), int(top_y)]

    def _the_thread(self, inPs, outPs):
        """Read the image from input stream, process it and send lane information

        Parameters
        ----------
        inPs : list(Pipe)
        0 - video frames
        outPs : list(Pipe)
        0 - array of left and right lane information
        """
        while True:
            try:
                #print("LANE DETECT LOG, STARTED THREAD")
                #  ----- read the input streams ----------
                stamps, image_in = inPs[0].recv()
                #print("LANE DETECT LOG, GOTTEN IMAGE")
                # proncess input frame and return array [left lane coeffs, right lane coeffs]
                lanes_coefficients,preprocessed_img = self.laneDetection(image_in)
                #print("LANE DETECT LOG, GOTTEN COEFFS", lanes_coefficients)

                stamp = time.time()
                #for outP in self.outPs:
                outPs[0].send([[stamp], lanes_coefficients])
                left_lane_pts = self.points_from_lane_coeffs(lanes_coefficients[0])
                right_lane_pts = self.points_from_lane_coeffs(lanes_coefficients[1])
                outPs[1].send([[stamp], [left_lane_pts, right_lane_pts]])
                outPs[2].send([[stamp], preprocessed_img])

            except Exception as e:
                print("LaneDetector failed to obtain lanes:",e,"\n")
                pass





