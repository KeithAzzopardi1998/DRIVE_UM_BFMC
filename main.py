import sys
sys.path.append('.')

import time
import signal
from multiprocessing import Pipe, Process, Event

from src.utils.remoteControlReceiver                import RemoteControlReceiver
from src.utils.cameraStreamer                       import CameraStreamer
from src.utils.cameraSpoofer                        import CameraSpoofer
from src.utils.cameraSpoofer                        import CameraSpoofer
from src.utils.perceptionVisualizer                 import PerceptionVisualizer
from src.hardware.camera.cameraProcess              import CameraProcess
from src.hardware.nucleoInterface.nucleoInterface   import NucleoInterface
#from src.control.autonomousController               import AutonomousController
from src.perception.laneDetection.laneDetector      import LaneDetector
#from src.perception.objectDetection.objectDetector  import ObjectDetector

# =============================== CONFIG =================================================
enableStream            =  True
enableCameraSpoof       =  False
enableRc                =  True
enableVisualization     =  True
enableLaneDetection     =  True
enableObjectDetection   =  False

#================================ PIPES ==================================================
laneR,  laneS   = Pipe(duplex = False)  # lane detection data
objR,   objS    = Pipe(duplex = False)  # object detection data
camR,   camS    = Pipe(duplex = False)  # video frame stream (from real/spoof camera)
nucR,   nucS    = Pipe(duplex = False)  # Nucleo commands (from autonomous/remote controller)

#================================ PROCESSES ==============================================
allProcesses = list()

# =============================== CAMERA =================================================
if enableStream: # set up stream

    if enableCameraSpoof: # use spoof camera
        camProc = CameraSpoofer([],[camS],'vid')
    else:                 # use real camera
        camProc = CameraProcess([],[camS])
    allProcesses.append(camProc)

    if enableVisualization:
        # set up intermediary process to visualize lane and object detection
        visR, visS = Pipe(duplex = False)
        visProc = PerceptionVisualizer([camR, laneR, objR], [visS],
            activate_ld=enableLaneDetection,
            activate_od=enableObjectDetection)
        allProcesses.append(visProc)
        streamProc = CameraStreamer([visR], [])
    else:
        # pipe camera feed directly to the streamer
        streamProc = CameraStreamer([camR], [])
    allProcesses.append(streamProc)

# =============================== PERCEPTION =============================================
# -------- Lane Detection -----------
if enableLaneDetection:
    laneProc = LaneDetector([camR], [laneS])
    allProcesses.append(laneProc)
# -------- Object Detection ---------
if enableObjectDetection:
    objProc = ObjectDetector([camR], [objS])
    allProcesses.append(objProc)

# =============================== CONTROL ================================================
if enableRc: # use romote controller
    conProc = RemoteControlReceiver([],[nucS])
else:        # use autonomous controller
    conProc = AutonomousController([laneR, objR],[nucS])
allProcesses.append(conProc)

nucProc = NucleoInterface([nucR], [])
allProcesses.append(nucProc)

# ========================================================================================

print("Starting the processes!",allProcesses)
for proc in allProcesses:
    proc.daemon = True
    proc.start()

blocker = Event()

try:
    blocker.wait()
except KeyboardInterrupt:
    print("\nCatching a KeyboardInterruption exception! Shutdown all processes.\n")
    for proc in allProcesses:
        if hasattr(proc,'stop') and callable(getattr(proc,'stop')):
            print("Process with stop",proc)
            proc.stop()
            proc.join()
        else:
            print("Process witouth stop",proc)
            proc.terminate()
            proc.join()
