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
from src.control.autonomousController               import AutonomousController
from src.perception.laneDetection.laneDetector      import LaneDetector
from src.perception.objectDetection.objectDetector  import ObjectDetector

# =============================== CONFIG =================================================
enableStream            =  True
enableCameraSpoof       =  False
enableRc                =  False
enableVisualization     =  True
enableLaneDetection     =  True
enableObjectDetection   =  True

#================================ PIPES ==================================================
laneR1,  laneS1   = Pipe(duplex = False)  # lane detection data (for visualization)
laneR2,  laneS2   = Pipe(duplex = False)  # lane detection data (for autonomous control)
objR1,   objS1    = Pipe(duplex = False)  # object detection data (for visualization)
objR2,   objS2    = Pipe(duplex = False)  # object detection data (for autonomous control)
camR1,   camS1    = Pipe(duplex = False)  # video frame stream (for visualization/streaming)
camR2,   camS2    = Pipe(duplex = False)  # video frame stream (for lane detection)
camR3,   camS3    = Pipe(duplex = False)  # video frame stream (for object detection)
nucR,   nucS    = Pipe(duplex = False)  # Nucleo commands (from autonomous/remote controller)
visOtherR, visOtherS = Pipe(duplex=False)

#================================ PROCESSES ==============================================
allProcesses = list()

# =============================== CAMERA =================================================
if enableStream: # set up stream

    if enableCameraSpoof: # use spoof camera
        camProc = CameraSpoofer([],[camS1, camS2, camS3],'vid')
    else:                 # use real camera
        camProc = CameraProcess([],[camS1, camS2, camS3])
    allProcesses.append(camProc)

    if enableVisualization:
        # set up intermediary process to visualize lane and object detection
        visR, visS = Pipe(duplex = False)
        visProc = PerceptionVisualizer([camR1, laneR1, objR1, visOtherR], [visS],
            activate_ld=enableLaneDetection,
            activate_od=enableObjectDetection)
        allProcesses.append(visProc)
        streamProc = CameraStreamer([visR], [])
    else:
        # pipe camera feed directly to the streamer
        streamProc = CameraStreamer([camR1], [])
    allProcesses.append(streamProc)

# =============================== PERCEPTION =============================================
# -------- Lane Detection -----------
if enableLaneDetection:
    laneProc = LaneDetector([camR2], [laneS1, laneS2, visOtherS])
    allProcesses.append(laneProc)
# -------- Object Detection ---------
if enableObjectDetection:
    objProc = ObjectDetector([camR3], [objS1, objS2])
    allProcesses.append(objProc)

# =============================== CONTROL ================================================
if enableRc: # use romote controller
    conProc = RemoteControlReceiver([],[nucS])
else:        # use autonomous controller
    conProc = AutonomousController([laneR2, objR2],[nucS])
    allProcesses.append(conProc)

#conProc = RemoteControlReceiver([],[nucS])
#allProcesses.append(conProc)
#conProc1 = AutonomousController([laneR2, objR2],[])
#allProcesses.append(conProc1)

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
