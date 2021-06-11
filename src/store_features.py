#!/usr/bin/env python3
""" 
Reads 2D ground truth from JRDB in kitti label format, runs align re-ID features and stores 3D+2D features separately for each of the groundtruth label. 
The features along with the whole grountruth label is stored in a file which can be later used to train DNNs for different tasks. 
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json 
import sys 
import yaml
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
        output_dir = rospy.get_param('~output_dir', '/media/ruthz/DATA/features')
        if(os.path.exists(output_dir+"/"+self.sequence)):
            rospy.logerr("!!sequence already played!!")
            exit()
        else:
            os.mkdir(output_dir+"/"+self.sequence)
            self.features3d_base = output_dir+"/"+self.sequence+"/features_3d"
            self.features2d_base = output_dir+"/"+self.sequence+"/features_2d"
            os.mkdir(self.features2d_base)
            os.mkdir(self.features3d_base)

        
        self.timestamp_pub = rospy.Publisher('/timestamp_publisher', Clock, queue_size=15)
        self.gt_boundingbox_pub = rospy.Publisher('/omni_yolo_bboxes', BoundingBoxes, queue_size=15)##remap this in launch file 

        self.timestamp_sub = message_filters.Subscriber('/timestamp_publisher', Clock, queue_size=10)
        self.detection_2d_sub = message_filters.Subscriber("detection2d_with_feature", detection2d_with_feature_array, queue_size=3)
        self.detection_3d_sub = message_filters.Subscriber("detection3d_with_feature", detection3d_with_feature_array, queue_size=3)

        self.timestamp_idx = 0 
        
        # * define msgs synchronizer based on time stamps. Gets 3D and 2D features and stores to files  
        ts = message_filters.ApproximateTimeSynchronizer([self.detection_2d_sub, self.detection_3d_sub, self.timestamp_sub], 10,2.0, allow_headerless=True) 
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

        rate = rospy.Rate(50) # 30hz

        if not rospy.core.is_initialized():
            raise rospy.exceptions.ROSInitException("client code must call rospy.init_node() first")
        try:
            while not rospy.core.is_shutdown():
                self.TimeStampPublisher()
                rate.sleep()
        except KeyboardInterrupt:
            rospy.core.signal_shutdown('keyboard interrupt')


    
    def StampSynchCallback(self, yolo_det_feature_msg, fpoint_det_feature_msg, timestamp_msg):
        # * saves features according timestamp from labels
        current_file_name = self.timestamps[self.timestamp_idx][0]
        current_file_name = current_file_name.split('.')[0]
        current_file_name = current_file_name.split('/')[-1]
        features3d_file  = self.features3d_base+"/"+current_file_name+".yaml"
        features2d_file = self.features2d_base+"/"+current_file_name+".yaml"
        features2d_dict = dict(); idx2dset = set()
        features3d_dict = dict(); idx3dset = set()
        for feature2d in yolo_det_feature_msg.detection2d_with_features:
            if feature2d.frame_det_id not in idx2dset:
                idx2d_ = feature2d.frame_det_id
                features2d_dict[idx2d_] = dict()
                features2d_dict[idx2d_]['xmin'] = feature2d.x1; features2d_dict[idx2d_]['ymin'] = feature2d.y1
                features2d_dict[idx2d_]['xmax'] = feature2d.x2; features2d_dict[idx2d_]['ymax'] = feature2d.y2
                feature_str = [str(element) for element in feature2d.feature]
                features2d_dict[idx2d_]['feature'] = ",".join(feature_str)
                idx2dset.add(idx2d_)
        
        for feature3d in fpoint_det_feature_msg.detection3d_with_features:
            if feature3d.frame_det_id not in idx3dset:
                #print(feature3d.frame_det_id)
                idx3d_ = feature3d.frame_det_id
                features3d_dict[idx3d_] = dict()
                ##this is fpoint net detection 
                features3d_dict[idx3d_]['x'] = feature3d.x; features3d_dict[idx3d_]['y'] = feature3d.y ; features3d_dict[idx3d_]['z'] = feature3d.z
                features3d_dict[idx3d_]['l'] = feature3d.l; features3d_dict[idx3d_]['h'] = feature3d.h; features3d_dict[idx3d_]['w'] = feature3d.w
                features3d_dict[idx3d_]['theta'] = feature3d.theta
                feature_str = [str(element) for element in feature3d.feature]
                features3d_dict[idx3d_]['feature'] = ",".join(feature_str)
                idx3dset.add(idx3d_) 
        with open(features3d_file, 'w') as outfile:
            yaml.dump(features3d_dict, outfile, default_flow_style=False)
        with open(features2d_file, 'w') as outfile:
            yaml.dump(features2d_dict, outfile, default_flow_style=False)            
        self.timestamp_idx += 1
        if(self.timestamp_idx == len(self.timestamps)):
            rospy.loginfo("Got Features for all time stamps provided in the groundtruth labels")
        print(self.timestamp_idx, len(self.timestamps))
        return 


 
    def TimeStampPublisher(self,):
        t = rospy.Time.from_sec(self.timestamps[self.timestamp_idx][1])
        time_msg = Clock()
        time_msg.clock = t 
        info_str = "publishing index %d" % self.timestamp_idx
        rospy.loginfo(info_str+" stamp= "+str(self.timestamps[self.timestamp_idx][1])) 
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

    def StoreFeatures(self, file_name:str, features:dict):
        return


    def cleanup(self,):
        print("Shutting down features storing node")




def main(args):       
    try:
        instance = StoreFeaturesWithGT()
    except KeyboardInterrupt:
        print("Shutting down features storing node.")


        

if __name__ == '__main__':
    main(sys.argv)