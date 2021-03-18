import json
import socket
import cv2
from threading       import Thread
import numpy as np
import time

from src.utils.templates.workerProcess import WorkerProcess

import traceback

class PerceptionVisualizer(WorkerProcess):
    # ===================================== INIT =========================================
    def __init__(self, inPs, outPs, activate_ld=True, activate_od=True):
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
        self.height, self.width, self.channels = self.imgSize
        self.activate_ld = activate_ld
        self.activate_od = activate_od
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
        readTh = Thread(name = 'ProcessStream', target = self._process_stream, args = (self.inPs, self.outPs, self.activate_ld, self.activate_od, ))
        self.threads.append(readTh)

    def _process_stream(self, inPs, outPs, activate_ld, activate_od):
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
                #print("LOG: received image")
                if activate_ld:
                    stamps, coefficients_numpy = inPs[1].recv() # every packet received should be a list with the left and right lane info
                    stamps, other_image = inPs[3].recv() # every packet received should be the segmented image from the lane detection
                if activate_od:
                    stamps, objects = inPs[2].recv()

                # ----- draw the lane lines --------------
                image_ld = self.getImage_ld(image_in, coefficients_numpy) if activate_ld else image_in

                # ----- draw the object bounding boxes ---
                image_od = self.getImage_od(image_in, objects) if activate_od else image_in

                # ----- we can add another frame here ----
                image_xy = other_image

                # ----- combine the images ---------------
                image_in_resized = cv2.resize(image_in,(int(self.width/2),int(self.height/2)))
                image_xy_resized = cv2.resize(image_xy,(int(self.width/2),int(self.height/2)))
                image_ld_resized = cv2.resize(image_ld,(int(self.width/2),int(self.height/2)))
                image_od_resized = cv2.resize(image_od,(int(self.width/2),int(self.height/2)))

                image_out = np.vstack((
                    np.hstack((image_in_resized,image_ld_resized)),
                    np.hstack((image_od_resized,image_xy_resized))
                ))

                # https://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html
                font = cv2.FONT_HERSHEY_SIMPLEX
                title_ld = "Lanes" if activate_ld else "Lanes - Disabled"
                title_od = "Objects" if activate_od else "Objects - Disabled"
                cv2.putText(image_out,'Raw',       (0,int(self.height*0.49)),                     font,1,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(image_out,title_ld,     (int(self.width*0.5),int(self.height*0.49)),   font,1,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(image_out,title_od,   (0,int(self.height*0.99)),                     font,1,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(image_out,'Other',     (int(self.width*0.5),int(self.height*0.99)),   font,1,(255,255,255),2,cv2.LINE_AA)

                # ----- write to the output stream -------
                assert image_out.shape == self.imgSize
                stamp = time.time()
                for outP in self.outPs:
                    outP.send([[stamp], image_out])


            except Exception as e:
                print("PerceptionVisualizer failed to process image:",e,"\n")
                traceback.print_exc()
                pass

    # ===================================== LANE DETECTION ===============================
    def get_vertices_for_img(self,img):
        img_shape = img.shape
        height = img_shape[0]
        width = img_shape[1]

        region_top_left     = (0.00*width, 0.50*height)
        region_top_right    = (1.00*width, 0.50*height)
        region_bottom_left  = (0.00*width, 1.00*height)
        region_bottom_right = (1.00*width, 1.00*height)

        vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)
        return vert

    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=10, make_copy=True):
        # Copy the passed image
        img_copy = np.copy(img) if make_copy else img

        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)

        return img_copy

    def trace_lane_line_with_coefficients(self, img, line_coefficients, top_y, make_copy=True):
        A = line_coefficients[0]
        b = line_coefficients[1]
        if A==0.0 and b==0.0:
            img_copy = np.copy(img) if make_copy else img
            return img_copy

        height, width,_ = img.shape
        bottom_y = height - 1
        # y = Ax + b, therefore x = (y - b) / A
        bottom_x = (bottom_y - b) / A
        # clipping the x values
        bottom_x = min(bottom_x, 2*width)
        bottom_x = max(bottom_x, -1*width)

        top_x = (top_y - b) / A
        # clipping the x values
        top_x = min(top_x, 2*width)
        top_x = max(top_x, -1*width)

        new_lines = [[[int(bottom_x), int(bottom_y), int(top_x), int(top_y)]]]
        return self.draw_lines(img, new_lines, make_copy=make_copy)

    def getImage_ld(self, image_in, coefficients_numpy):
        img = image_in.copy()
        vert = self.get_vertices_for_img(img)
        left_coefficients = coefficients_numpy[0]
        right_coefficients = coefficients_numpy[1]
        region_top_left = vert[0][1]

        lane_img_left = self.trace_lane_line_with_coefficients(img, left_coefficients, region_top_left[1], make_copy=True)
        lane_img_both = self.trace_lane_line_with_coefficients(lane_img_left, right_coefficients, region_top_left[1], make_copy=False)

        # image1 * α + image2 * β + λ
        # image1 and image2 must be the same shape.
        img_with_lane_weight =  cv2.addWeighted(img, 0.7, lane_img_both, 0.3, 0.0)

        return img_with_lane_weight

    # ===================================== OBJECT DETECTION =============================
    def getImage_od(self, image_in, object_list):
        # based on https://github.com/google-coral/pycoral/blob/master/examples/detect_image.py
        image = image_in.copy()

        #print(object_list)

        if not object_list:
            #print('list was empty')
            return image
        else:
            for obj in object_list:
                #print(obj)
                w = image.shape[1]
                h = image.shape[0]
                ymin, xmin, ymax, xmax = obj['bounding_box']
                xmin = int(xmin * w)
                xmax = int(xmax * w)
                ymin = int(ymin * h)
                ymax = int(ymax * h)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                y = ymin - 15 if ymin - 15 > 15 else ymin + 15
                cv2.putText(image,"{}: {:.2f}%".format(self.LABEL_DICT[obj['class_id']], obj['score'] * 100),
                            (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        return image
