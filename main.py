import os
from queue import Queue
from vbm import RgbdCam
from vbm import Manipulator

if __name__ == "__main__":
    q = Queue()
    model_directory = os.environ['HOME'] + '/yolov8_rs/yolov8m.pt'
    cam = RgbdCam(object_name="scissors",queue=q, model_directory = model_directory)
    robot = Manipulator(queue=q)
    cam.start()
    robot.start()