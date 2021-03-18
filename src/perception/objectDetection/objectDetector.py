import numpy as np
import cv2
import time
import math
from threading import Thread
from collections import deque
import src.perception.laneDetection.preProcessing as pp

from src.utils.templates.workerProcess import WorkerProcess

import tflite_runtime.interpreter as tflite

import platform

_EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]

def load_edgetpu_delegate(options=None):
    """Loads the Edge TPU delegate with the given options."""
    return tflite.load_delegate(_EDGETPU_SHARED_LIB, options or {})

def make_interpreter(model_path_or_content, device=None):
    """Returns a new interpreter instance.

    Interpreter is created from either model path or model content and attached
    to an Edge TPU device.

    Args:
        model_path_or_content (str or bytes): `str` object is interpreted as
        model path, `bytes` object is interpreted as model content.
        device (str): The type of Edge TPU device you want:

        + None      -- use any Edge TPU
        + ":<N>"    -- use N-th Edge TPU
        + "usb"     -- use any USB Edge TPU
        + "usb:<N>" -- use N-th USB Edge TPU
        + "pci"     -- use any PCIe Edge TPU
        + "pci:<N>" -- use N-th PCIe Edge TPU

    Returns:
        New ``tf.lite.Interpreter`` instance.
    """
    delegates = [load_edgetpu_delegate({'device': device} if device else {})]
    if isinstance(model_path_or_content, bytes):
        return tflite.Interpreter(model_content=model_path_or_content, experimental_delegates=delegates)
    else:
        return tflite.Interpreter(model_path=model_path_or_content, experimental_delegates=delegates)

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
        #Object Detection interpreter
        self.od_interpreter = tflite.Interpreter(model_path='./models/object_detector_quant_4.tflite')
        self.od_interpreter.allocate_tensors()
        self.od_input_details = self.od_interpreter.get_input_details()
        self.threshold = 0.3

        #Traffic Sign Recognition interpreter
        self.tsr_interpreter = tflite.Interpreter(model_path='./models/object_recognition_quant.tflite')
        self.tsr_interpreter.allocate_tensors()
        self.tsr_input_details = self.tsr_interpreter.get_input_details()
        self.tsr_output_details = self.tsr_interpreter.get_output_details()
        self.threshold = 0.3
        #print("finished setting up  models")
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

    #======================= TRAFFIC SIGN RECOGNITION=======

    #======================= METHODS =======================
    def set_input_od(self, image):
        """Sets the input tensor."""
        tensor_index = self.od_input_details[0]['index']
        input_tensor = self.od_interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image
    
    def set_input_tsr(self, image):
        """Sets the input tensor."""
        tensor_index = self.tsr_input_details[0]['index']
        input_tensor = self.tsr_interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def get_output_tensor(self, index):
        """Returns the output tensor at the given index."""
        output_details = self.od_interpreter.get_output_details()[index]
        tensor = np.squeeze(self.od_interpreter.get_tensor(output_details['index']))
        return tensor

    def make_result(self, box, class_id, scores):
        result = {
                    'bounding_box': box,
                    'class_id': class_id,
                    'score': scores
        }
        return result

    def objectDetection(self, img_in):
        input_shape = self.od_input_details[0]['shape']
        _, height, width, _ = input_shape

        resized_image = cv2.resize(img_in, (width, height), interpolation=cv2.INTER_LINEAR)

        resized_image = resized_image[np.newaxis, :]

        self.set_input_od(resized_image)

        self.od_interpreter.invoke()

        boxes = np.clip(self.get_output_tensor(0), 0, 1)
        classes = self.get_output_tensor(1)
        scores = self.get_output_tensor(2)
        count = int(self.get_output_tensor(3))

        results = [self.make_result(boxes[i], classes[i], scores[i]) for i in range(count) if scores[i] >= self.threshold]

        #print(results)

        return results

    def signRecognition(self,img_in):
        input_shape = self.tsr_input_details[0]['shape']
        _, height, width, _ = input_shape

        resized_image = cv2.resize(img_in, (width, height), interpolation=cv2.INTER_LINEAR)

        resized_image = resized_image[np.newaxis, :]

        self.set_input_tsr(resized_image)

        self.tsr_interpreter.invoke()

        output_proba = self.tsr_interpreter.get_tensor(self.tsr_output_details[0]['index'])[0]
        tsr_class = np.argmax(output_proba)

        return float("7.%d"%tsr_class)

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
                #("LOG: going to run OD model")
                obj_list = self.objectDetection(image_in)

                #looping through the list of objects, and updating
                #the class ID of any traffic signs
                for o in obj_list:
                    #the main OD model uses class 7 for traffic signs
                    if o['class_id']== 7.0:
                        #grab the part of the image containing the sign
                        w = image_in.shape[1]
                        h = image_in.shape[0]
                        ymin, xmin, ymax, xmax = o['bounding_box']
                        xmin = int(xmin * w)
                        xmax = int(xmax * w)
                        ymin = int(ymin * h)
                        ymax = int(ymax * h)
                        roi = image_in[ymin:ymax, xmin:xmax]
                        #run the traffic sign recognition function,
                        #which returns the new class ID
                        o['class_id'] = self.signRecognition(roi)

                #print("LOG: detected %d objects"%len(obj_list))
                stamp = time.time()
                for outP in self.outPs:
                    outP.send([[stamp], obj_list])

            except Exception as e:
                print("ObjectDetector failed to obtain objects:",e,"\n")
                pass





