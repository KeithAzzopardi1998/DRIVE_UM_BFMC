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
        #self.interpreter = tflite.Interpreter(model_path='./models/object_detector_quant_4_edgetpu.tflite',
        #experimental_delegates=[tflite.load_delegate('libedgetpu.so.1.0')])
        self.interpreter = tflite.Interpreter(model_path='./models/object_detector_quant_4.tflite')
        #self.interpreter = make_interpreter('./models/object_detector_quant_4_edgetpu.tflite', device='usb')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.threshold = 0.3
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
    def set_input_tensor(self, image):
        """Sets the input tensor."""
        tensor_index = self.input_details[0]['index']
        input_tensor = self.interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def get_output_tensor(self, index):
        """Returns the output tensor at the given index."""
        output_details = self.interpreter.get_output_details()[index]
        tensor = np.squeeze(self.interpreter.get_tensor(output_details['index']))
        return tensor

    def make_result(self, box, class_id, scores):
        result = {
                    'bounding_box': box,
                    'class_id': class_id,
                    'score': scores
        }
        return result

    def objectDetection(self, img_in):
        input_shape = self.input_details[0]['shape']
        _, height, width, _ = input_shape

        resized_image = cv2.resize(img_in, (width, height), interpolation=cv2.INTER_LINEAR)

        resized_image = resized_image[np.newaxis, :]

        self.set_input_tensor(resized_image)

        self.interpreter.invoke()

        boxes = np.clip(self.get_output_tensor(0), 0, 1)
        classes = self.get_output_tensor(1)
        scores = self.get_output_tensor(2)
        count = int(self.get_output_tensor(3))

        results = [self.make_result(boxes[i], classes[i], scores[i]) for i in range(count) if scores[i] >= self.threshold]

        print(results)

        return results


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





