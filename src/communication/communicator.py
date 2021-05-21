from src.utils.templates.workerProcess import WorkerProcess
from threading import Thread
import math
import time
import numpy as np

class Communicator(WorkerProcess):
    #======================= INIT =======================
    def __init__(self, inPs, outPs):
        """Accepts the detected items from the visualization
        module, and instructs nucleo
        ------------
        inPs : list(Pipe)
            0 - lanes
            1 - objects
        outPs : list(Pipe)
            0 - high-level instructions
        """
        super(Communicator,self).__init__(inPs, outPs)

        #load list of nodes

    #======================= RUN =======================
    def run(self):
        """Apply the initializing methods and start the threads.
        """
        super(Communicator,self).run()

    #======================= INIT THREADS =======================
    def _init_threads(self):

        thr = Thread(name='CommunicatorThread',target = self._comms_thread, args= (self.inPs, self.outPs, ))
        thr.daemon = True
        self.threads.append(thr)

    #======================= MAIN THREAD =======================
    def _comms_thread(self, inPs, outPs):
        """Communicate with infrastructure and pass on messages to other processes

        Parameters
        ----------
        inPs : list(Pipe)
        outPs : list(Pipe)
            0 - current node
            1 - tuple of TL states (TODO)
        """

        while True:
            try:
                node_id = self.getNodeFromGps()

                stamp = time.time()
                outPs[0].send([[stamp],node_id])

            except Exception as e:
                print("Communicator failed:",e,"\n")
                pass
    #======================= AUXILIARY FUNCTIONS =======================

    #listen for GPS data packet and determine current node
    def getNodeFromGps():
        return 0

    #TODO listen for semaphore data packet and determine TL states
     