import numpy as np
import cv2
import time
import math
from threading import Thread
from collections import deque
from scipy import stats
import src.perception.laneDetection.preProcessing as pp
from matplotlib import pyplot as plt

from src.utils.templates.workerProcess import WorkerProcess


from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

class ObjectDetector(WorkerProcess):
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
        self.interpreter = make_interpreter('./models/object_detector_quant_4_edgetpu.tflite')
        self.interpreter.allocate_tensors()
        self.threshold=0.2
        print("finished setting up OD model")
        super(ObjectDetector,self).__init__(inPs, outPs)
        
    #======================= RUN =======================
    def run(self):
        """Apply the initializing methods and start the threads.
        """
        super(ObjectDetector,self).run()

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
    def objectDetection(self, img_in):
        image = Image.fromarray(np.uint8(img_in)).convert('RGB')
        #plt.imshow(image)
        _, scale = common.set_resized_input(
            self.interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
        self.interpreter.invoke()
        return detect.get_objects(interpreter, self.threshold, scale)


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
                #  ----- read the input streams ----------
                stamps, image_in = inPs[0].recv()
                print("LOG: going to run OD model")
                obj_list = self.objectDetection(image_in)
                print("LOG: detected %d objects"%len(obj_list))
                stamp = time.time()
                for outP in self.outPs:
                    outP.send([[stamp], obj_list])

            except Exception as e:
                print("ObjectDetector failed to obtain objects:",e,"\n")
                pass





