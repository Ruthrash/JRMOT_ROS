
cd /home/ruthz/cvpr_challenge/JRDB/kitti_labels
for d in * 
do 
roslaunch jpda_rospack store_features.launch sequence_name:="$d" 

done
