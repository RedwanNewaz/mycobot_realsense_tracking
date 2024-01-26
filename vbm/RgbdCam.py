import os
import cv2
import time
import math
import numpy as np
from queue import Queue
import pyrealsense2 as rs
from ultralytics import YOLO
from threading import Thread


class RgbdCam(Thread):
    def __init__(self,object_name, queue,model_directory):
        W = 640
        H = 480

        config = rs.config()
        config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)

        self.pipeline = rs.pipeline()
        profile = self.pipeline.start(config)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        
        self.model = YOLO(model_directory)
        self.objName = object_name
        self.queue = queue
        

        Thread.__init__(self)

    
    def transform_to_world_coordinates(self,x, y, depth, intrin):
    # Convert pixel coordinates to 3D world coordinates
        pixel = np.array([x, y], dtype=np.float)
        depth_point = rs.rs2_deproject_pixel_to_point(intrin, pixel, depth)

        return depth_point

    def run(self):

        while True:
            #time1 = time.time()
            frames = self.pipeline.wait_for_frames()

            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

            results = self.model(color_image)
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0].to('cpu').detach().numpy().copy()  # get box coordinates in (top, left, bottom, right) format
                    c = box.cls
                    pixel_distance = depth_frame.get_distance(int((b[0]+b[2])/2) , int((b[1]+b[3])/2))

                    cv2.rectangle(depth_colormap, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255),
                                thickness = 2, lineType=cv2.LINE_4)
                    cv2.putText(depth_colormap, text = self.model.names[int(c)] + str(pixel_distance), org=(int(b[0]), int(b[1])),
                                fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.7, color = (0, 0, 255),
                                thickness = 2, lineType=cv2.LINE_4)
                    world_coordinates = self.transform_to_world_coordinates(int((b[0]+b[2])/2), int((b[1]+b[3])/2), pixel_distance, depth_intrin)
                    world_coordinates_mm = [1000 * world_coordinates[0],1000 * world_coordinates[1], 1000 * world_coordinates[2]]
                    print("class and World Coordinates (X, Y, Z):",self.model.names[int(c)], world_coordinates_mm)
                    if self.model.names[int(c)] == self.objName:

                        self.queue.put(world_coordinates_mm)
                    
                    
                    #print("pixel domain coordinate:",model.names[int(c)],"[",int((b[0]+b[2])/2),int((b[1]+b[3])/2),"]")
            annotated_frame = results[0].plot()

            cv2.imshow("color_image", annotated_frame)
            cv2.imshow("depth_image", depth_colormap)
            #time2 = time.time()
            #print(f"FPS : {int(1/(time2-time1))}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    q = Queue()
    model_directory = os.environ['HOME'] + '/yolov8_rs/yolov8m.pt'
    cam = RgbdCam(object_name="scissors",queue=q, model_directory = model_directory)

    cam.start()
    while True:
        if not q.empty():
            cord = q.get()
