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
	def laneDetection(self,img):
		
