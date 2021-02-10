import json
import socket
import cv2
from threading       import Thread
import numpy as np

from src.utils.templates.workerProcess import WorkerProcess

class PerceptionVisualizer(WorkerProcess):
    # ===================================== INIT =========================================
    def __init__(self, inPs, outPs):
        """Accepts frames from the camera, visualizes the results from the lane
        and object detection algorithms, and transmits a frame of the same size

        Parameters
        ------------
        inPs : list(Pipe)
            0 - video frames
            1 - detected lanes
            2 - detected objects
        outPs : list(Pipe) 
            List of output pipes (order does not matter)
        """
        super(PerceptionVisualizer,self).__init__(inPs, outPs)

        self.imgSize    = (480,640,3)
        self.LABEL_DICT = {0: 'bike',
                1: 'bus',
                2: 'car',
                3: 'motor',
                4: 'person',
                5: 'rider',
                6: 'traffic_light',
                7: 'traffic_sign',
                8: 'train',
                9: 'truck'
            }
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABEL_DICT), 3), dtype="uint8")
        
    # ===================================== RUN ==========================================
    def run(self):
        """ start the threads"""
        super(PerceptionVisualizer,self).run()
    
    # ===================================== INIT THREADS =================================
    def _init_threads(self):
        """Initialize the read thread to receive the video.
        """
        readTh = Thread(name = 'ProcessStream',target = self._process_stream, args = (self.inPs,self.outPs, ))
        self.threads.append(readTh)    

    def _process_stream(self, inPs, outPs):
        """Read the image from input stream, process it and send it
        over the output stream
        
        Parameters
        ----------
        inPs : list(Pipe)
            0 - video frames
            1 - detected lanes
            2 - detected objects
        outPs : list(Pipe) 
            List of output pipes (order does not matter)
        """
        while True:
            try:
                #  ----- read the input streams ---------- 
                stamps, image_in = inPs[0].recv()
                _ = inPs[1].recv()
                objects = inPs[2].recv()

                # ----- draw the lane lines --------------

                # ----- draw the object bounding boxes ---
                image_od = self.getImage_od(image_in, objects)
                # ----- combine the images ---------------

                # ----- write to the output stream -------

            except Exception as e:
                print("PerceptionVisualizer failed to process image:",e,"\n")
                pass       

    # ===================================== OBJECT DETECTION ===============================             
    def getImage_od(self, img_in, object_list):
        original_numpy = np.copy(img_in)
        for obj in object_list:
            # Convert the bounding box figures from relative coordinates
            # to absolute coordinates based on the original resolution
            ymin, xmin, ymax, xmax = obj['bounding_box']
            print(ymin, xmin, ymax, xmax)
            xmin = int(xmin * original_numpy.shape[1])
            xmax = int(xmax * original_numpy.shape[1])
            ymin = int(ymin * original_numpy.shape[0])
            ymax = int(ymax * original_numpy.shape[0])

            # Grab the class index for the current iteration
            idx = int(obj['class_id'])
            # Skip the background
            if idx >= len(self.LABEL_DICT):
                continue

            # draw the bounding box and label on the image
            color = [int(c) for c in self.COLORS[idx]]
            cv2.rectangle(original_numpy, (xmin, ymin), (xmax, ymax), 
                        color, 2)
            y = ymin - 15 if ymin - 15 > 15 else ymin + 15
            label = "{}: {:.2f}%".format(self.LABEL_DICT[obj['class_id']],
                obj['score'] * 100)
            cv2.putText(original_numpy, label, (xmin, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        img_out = (original_numpy * 255).astype(np.uint8)
        return img_out    