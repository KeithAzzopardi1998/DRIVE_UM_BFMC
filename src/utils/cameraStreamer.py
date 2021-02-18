import socket
import struct
import time
import numpy as np
from multiprocessing import Process
from threading import Thread

import cv2
import os

from src.utils.templates.workerProcess import WorkerProcess

class CameraStreamer(WorkerProcess):
    # ===================================== INIT =========================================
    def __init__(self, inPs, outPs):
        """Process used for sending images over the network. UDP protocol is used. The
        image is compressed before it is send. 

        Used for visualizing your raspicam from PC.
        
        Parameters
        ----------
        inPs : list(Pipe) 
            List of input pipes, only the first pipe is used to transfer the captured frames. 
        outPs : list(Pipe) 
            List of output pipes (not used at the moment)
        """
        super(CameraStreamer,self).__init__( inPs, outPs)

        self.serverIp   =  os.environ['IP_PC'] # PC ip
        self.port       =  2244            # com port
        
    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads.
        """
        self._init_socket()
        super(CameraStreamer,self).run()

    # ===================================== INIT THREADS =================================
    def _init_threads(self):
        """Initialize the sending thread.
        """
        if self._blocker.is_set():
            return
        streamTh = Thread(name='StreamSending',target = self._send_thread, args= (self.inPs[0], ))
        streamTh.daemon = True
        self.threads.append(streamTh)

    # ===================================== INIT SOCKET ==================================
    def _init_socket(self):
        """Initialize the socket. 
        """
        self.client_socket = socket.socket()
        self.connection = None
        # Trying repeatedly to connect the camera receiver.
        try:
            while self.connection is None and not self._blocker.is_set():
                try:
                    self.client_socket.connect((self.serverIp, self.port))
                    self.client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                    self.connection = self.client_socket.makefile('wb') 
                except ConnectionRefusedError as error:
                    time.sleep(0.5)
                    pass
        except KeyboardInterrupt:
            self._blocker.set()
            pass

        
    # ===================================== SEND THREAD ==================================
    def _send_thread(self, inP):
        """Sending the frames received thought the input pipe to remote client by using a socket. 
        
        Parameters
        ----------
        inP : Pipe
            Input pipe to read the frames from other process. 
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        print('Start streaming')

        while True:
            time.sleep(0.05)
            try:
                #print("LOG: fetching image")
                stamps, image = inP.recv()
                print("STREAMER LOG: got something from the input pipe ... image variable is of shape",image.shape)
                result, image = cv2.imencode('.jpg', image, encode_param)
                #print("LOG: finished encoding the image")
                data   =  image.tobytes()
                #print("LOG: finshed converting image to byte stream")
                size   =  len(data)
                #print("LOG: packaged image into packet of size %d ... going to transmit"%size)

                self.connection.write(struct.pack("<L",size))
                self.connection.write(data)
                #print("LOG: successfully transmitted frame")

            except Exception as e:
                print("CameraStreamer failed to stream images:",e,"\n")
                # Reinitialize the socket for reconnecting to client.  
                self.connection = None
                self._init_socket()
                pass
