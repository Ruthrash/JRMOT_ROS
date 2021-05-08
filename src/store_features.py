#!/usr/bin/env python3
""" 
Reads 2D ground truth from JRDB in kitti label format, runs align re-ID features and stores 3D+2D features separately for each of the groundtruth label. 
The features along with the whole grountruth label is stored in a file which can be later used to train DNNs for different tasks. 
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json 
import sys 
# *
import rospy
import numpy as np
import message_filters
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image  
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from jpda_rospack.msg import detection3d_with_feature_array, \
    detection3d_with_feature, detection2d_with_feature_array


# * synch image from ros topic and groundtruth with time stamps 
# * Using 2D bboxes from GT for the synched image get 3D and 2D features with time synch and store in a file along with GT.
class StoreFeaturesWithGT:
    def __init__(self,): 

        self.node_name = "store_features_w_gt"
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)
        self.timestamps_base_dir = "/home/ruthz/cvpr_challenge/JRDB/timestamps"
        self.labels_base_dir = "/home/ruthz/cvpr_challenge/JRDB/kitti_labels"
        self.sequence = rospy.get_param('~sequence_name', 'bytes-cafe-2019-02-07_0')
        self.output_dir = rospy.get_param('~output_dir', '/home')
        
        self.timestamp_pub = rospy.Publisher('/timestamp_publisher', Clock, queue_size=15)
        self.gt_boundingbox_pub = rospy.Publisher('/omni_yolo_bboxes', BoundingBoxes, queue_size=15)##remap this in launch file 

        self.timestamp_sub = message_filters.Subscriber('/timestamp_publisher', Clock)
        self.detection_2d_sub = message_filters.Subscriber("detection2d_with_feature", detection2d_with_feature_array, queue_size=5)
        self.detection_3d_sub = message_filters.Subscriber("detection3d_with_feature", detection3d_with_feature_array, queue_size=5)

        self.timestamp_idx = 0 
        
        # * define msgs synchronizer based on time stamps. Gets 3D and 2D features and stores to files  
        ts = message_filters.ApproximateTimeSynchronizer([self.detection_2d_sub, self.detection_3d_sub, self.timestamp_sub], 50, 0.5, allow_headerless=True) 
        ts.registerCallback(self.StampSynchCallback)

        #
        self.timestamps = []##check this 
        stamp_file = os.path.join(self.timestamps_base_dir, self.sequence+"/frames_img.json") 
        f = open(stamp_file)
        full_data = json.load(f)
        full_data = full_data['data']
        for idx in range(len(full_data)):
            for camera in full_data[idx]['cameras']:
                if camera['name'] == 'stitched_image0':
                    self.timestamps.append(tuple((camera['url'], camera['timestamp'])))

        rate = rospy.Rate(30) # 20hz
        while not rospy.is_shutdown():
            print("waiting for topics")
            self.TimeStampPublisher()
            rate.sleep()
        #print(self.timestamps)

    
    def StampSynchCallback(self, yolo_det_feature_msg, fpoint_det_feature_msg, timestamp_msg ):
        # * saves features to according timestamp or image file
        #print(fpoint_det_feature_msg)
        
        self.timestamp_idx += 1

        return 


 
    def TimeStampPublisher(self,):
        t = rospy.Time.from_sec(self.timestamps[self.timestamp_idx][1])
        time_msg = Clock()
        time_msg.clock = t 
        info_str = "publishing index %d" % self.timestamp_idx
        rospy.loginfo(info_str)
        label_file = self.timestamps[self.timestamp_idx][0].split('/')[-1]
        label_file = label_file.split('.')[0]
        label_file_path = self.labels_base_dir+"/"+self.sequence+"/"+label_file+".txt"
        bboxes_msg = self.GetBBoxesMsg(label_file_path)
        bboxes_msg.header.stamp = t
        #print(t)
        
        # also publish bounding boxes 

        self.timestamp_pub.publish(time_msg)
        self.gt_boundingbox_pub.publish(bboxes_msg)
    


    def GetBBoxesMsg(self, label_file:str):
        output_bboxes_msg = BoundingBoxes()
        with open(label_file,'r') as file:  
            id_ = 0  
            for line in file:
                gt = line.split() 
                bbox_msg = BoundingBox()
                #print(gt)
                bbox_msg.xmin = int(gt[5]); bbox_msg.ymin = int(gt[6])
                bbox_msg.xmax = int(gt[7]); bbox_msg.ymax = int(gt[8])
                bbox_msg.id = id_ ; bbox_msg.Class = gt[0]
                id_ += 1
                output_bboxes_msg.bounding_boxes.append(bbox_msg)
        return output_bboxes_msg


    def cleanup(self):
        print("Shutting down features storing node")




def main(args):       
    try:
        instance = StoreFeaturesWithGT()
    except KeyboardInterrupt:
        print("Shutting down features storing node.")


        

if __name__ == '__main__':
    main(sys.argv)