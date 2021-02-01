import serial
import time
from multiprocessing import Event

from src.utils.templates.workerProcess          import WorkerProcess
from src.hardware.nucleoInterface.fileHandler   import FileHandler
from src.hardware.nucleoInterface.readThread    import ReadThread
from src.hardware.nucleoInterface.writeThread   import WriteThread


class NucleoInterface(WorkerProcess):
    # ===================================== INIT =========================================
    def __init__(self,inPs, outPs):
        """The functionality of this process is to redirectionate the commands from the remote or other process to the micro-controller by serial port.
        The default frequency is 256000 and device file /dev/ttyACM0. It automatically save the sent commands into a log file, named historyFile.txt. 
        
        Parameters
        ----------
        inPs : list(Pipes)
            A list of pipes, where the first element is used for receiving the command to control the vehicle from other process.
        outPs : None
            Has no role.
        """
        super(NucleoInterface,self).__init__(inPs, outPs)

        devFile = '/dev/ttyACM0'
        logFile = 'historyFile.txt'
        
        # comm init       
        self.serialCom = serial.Serial(devFile,256000,timeout=0.1)
        self.serialCom.flushInput()
        self.serialCom.flushOutput()

        # log file init
        self.historyFile = FileHandler(logFile)
        
        

    # ===================================== INIT THREADS =================================
    def _init_threads(self):
        """ Initializes the read and the write thread.
        """
        # read write thread        
        readTh  = ReadThread(self.serialCom,self.historyFile)
        self.threads.append(readTh)
        writeTh = WriteThread(self.inPs[0], self.serialCom, self.historyFile)
        self.threads.append(writeTh)
    

    def run(self):
        super(NucleoInterface,self).run()
        #Post running process -> close the history file
        self.historyFile.close()
