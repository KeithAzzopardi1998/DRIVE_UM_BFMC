import numpy as np 
import cv2 
import math
from threading import Thread
from collections import deque
from scipy import stats
import preProcessing as pp 

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

        thr = Thread(name='StreamSending',target = self._the_thread, args= (self.inPs[0], self.outPs[0], ))
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
        img = image_in.copy()
        combined_hsl_img = pp.filter_img_hsl(img)
        grayscale_img = pp.grayscale(combined_hsl_img)
        gaussian_smoothed_img = pp.gaussian_blur(grayscale_img, kernel_size=5)
        canny_img = cv2.Canny(gaussian_smoothed_img, 50, 150)
        segmented_img = pp.getROI(canny_img)
        hough_lines = hough_transform(segmented_img, rho, theta, threshold, min_line_length, max_line_gap)

        try:
			left_lane_lines, right_lane_lines = pp.separate_lines(hough_lines, img)
			left_lane_slope, left_intercept = pp.getLanesFormula(left_lane_lines)
			right_lane_slope, right_intercept = pp.getLanesFormula(right_lane_lines)
			smoothed_left_lane_coefficients = pp.determine_line_coefficients(left_lane_coefficients, [left_lane_slope, left_intercept])
			smoothed_right_lane_coefficients = pp.determine_line_coefficients(right_lane_coefficients, [right_lane_slope, right_intercept])

			return np.array([smoothed_left_lane_coefficients, smoothed_right_lane_coefficients])

		except Exception as e:
			#print("*** Error - will use saved coefficients ", e)
			smoothed_left_lane_coefficients = pp.determine_line_coefficients(left_lane_coefficients, [0.0, 0.0])
			smoothed_right_lane_coefficients = pp.determine_line_coefficients(right_lane_coefficients, [0.0, 0.0])

			return np.array([smoothed_left_lane_coefficients, smoothed_right_lane_coefficients])

