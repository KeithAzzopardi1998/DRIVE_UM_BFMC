import sys
sys.path.append('.')

import time
import signal
from multiprocessing import Pipe, Process, Event 

from src.utils.remoteControlReceiver        import RemoteControlReceiver
from src.utils.cameraStreamer               import CameraStreamer
from src.utils.cameraSpoofer                import CameraSpoofer
from src.hardware.camera                    import CameraProcess
from src.nucleoInterface.nucleoInterface    import NucleoInterface

# =============================== CONFIG =================================================
enableStream        =  True
enableCameraSpoof   =  False 
enableRc            =  True
#================================ PIPES ==================================================


#================================ PROCESSES ==============================================
allProcesses = list()

# =============================== HARDWARE PROCC =========================================
# ------------------- camera + streamer ----------------------
if enableStream:
    camStR, camStS = Pipe(duplex = False)           # camera  ->  streamer

    if enableCameraSpoof:
        camSpoofer = CameraSpoofer([],[camStS],'vid')
        allProcesses.append(camSpoofer)

    else:
        camProc = CameraProcess([],[camStS])
        allProcesses.append(camProc)

    streamProc = CameraStreamer([camStR], [])
    allProcesses.append(streamProc)



# ===================================== CONTROL ==========================================
#------------------- remote controller -----------------------
if enableRc:
    rcShR, rcShS   = Pipe(duplex = False)           # rc      ->  serial handler

    # serial handler process
    shProc = NucleoInterface([rcShR], [])
    allProcesses.append(shProc)

    rcProc = RemoteControlReceiver([],[rcShS])
    allProcesses.append(rcProc)

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
