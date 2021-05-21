import numpy as np
import cv2
import time
import math
from threading import Thread
from collections import deque
import src.perception.laneDetection.preProcessing as pp
import time

from src.utils.templates.workerThread import WorkerThread

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

class ObjectDetector(Thread):
    #======================= INIT =======================
    def __init__(self,inQs):
        """Accepts frames from the camera, processes the frame and detect lanes,
        and transmits information about left and right lanes.
        Parameters
        ------------
        inPs : list(Pipe)
            0 - receive image feed from the camera
        outPs : list(Pipe)
            0 - send lane information
        """
        super(ObjectDetector,self).__init__()
        
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

        self.inQs=inQs
        self.daemon = True
        #print("finished setting up  models")
        
        
    #======================= RUN =======================
    def run(self):
        """Apply the initializing methods and start the threads.
        """
        while True:
            time.sleep(1)
            print("OD: queue size is ",self.inQs[0].qsize())
            print("OD: queue object id is ",id(self.inQs[0]))
            if not self.inQs[0].empty():
                try:
                    print("started an object detection loop")
                    #  ----- read the input streams ----------
                    # TODO fix this
                    image_in = self.inQs[0].get()

                    #image_brightened = self.increase_brightness(image_in, value=30)
                    
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

                    print("LOG: detected %d objects"%len(obj_list))
                    

                except Exception as e:
                    print("ObjectDetector failed to obtain objects:",e,"\n")
                    pass
            else:
                print("no image in the queue")


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

    def increase_brightness(self, img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

        return img






