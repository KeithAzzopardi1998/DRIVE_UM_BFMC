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
        segmented_img = pp.getROI(canny_img)
        #print("LANE LINES LOG - SEGMENTED IMAGE SUM", np.sum(segmented_img))
        hough_lines = pp.hough_transform(segmented_img, rho, theta, threshold, min_line_length, max_line_gap)
        #print("LANE LINES LOG - HOUGH LINES", hough_lines)

        preprocessed_img = cv2.cvtColor(segmented_img,cv2.COLOR_GRAY2BGR)

        try:
            left_lane_lines, right_lane_lines = pp.separate_lines(hough_lines, img)
            left_lane_slope, left_intercept = pp.getLanesFormula(left_lane_lines)
            right_lane_slope, right_intercept = pp.getLanesFormula(right_lane_lines)
            smoothed_left_lane_coefficients = pp.determine_line_coefficients(left_lane_coefficients, [left_lane_slope, left_intercept])
            smoothed_right_lane_coefficients = pp.determine_line_coefficients(right_lane_coefficients, [right_lane_slope, right_intercept])
            
            #print("LANE LINES LOG - LEFT:", smoothed_left_lane_coefficients)

            #print("LANE LINES LOG - RIGHT:", smoothed_right_lane_coefficients)
            
            return np.array([smoothed_left_lane_coefficients, smoothed_right_lane_coefficients]), preprocessed_img

        except Exception as e:
            #print("*** Error - will use saved coefficients ", e)
            smoothed_left_lane_coefficients = pp.determine_line_coefficients(left_lane_coefficients, [0.0, 0.0])
            smoothed_right_lane_coefficients = pp.determine_line_coefficients(right_lane_coefficients, [0.0, 0.0])

            return np.array([smoothed_left_lane_coefficients, smoothed_right_lane_coefficients]), preprocessed_img

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
                outPs[1].send([[stamp], preprocessed_img])

            except Exception as e:
                print("LaneDetector failed to obtain lanes:",e,"\n")
                pass





