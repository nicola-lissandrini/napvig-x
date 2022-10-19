#ifndef ROS_CONVERSIONS_H
#define ROS_CONVERSIONS_H

#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Quaternion.h>
#include <torch/all.h>
#include <lietorch/pose.h>

#define LASER_SCAN_DIM 2

void laserScanToTensor (const sensor_msgs::LaserScan &scanMsg, torch::Tensor &tensorMsg);
void pose2ToPoseMsg (const lietorch::Pose2 &pose, geometry_msgs::Pose &poseMsg);

void pointMsgToTorch (const geometry_msgs::Point &vectorMsg, torch::Tensor &tensor);
void quaternionMsgToTorch (const geometry_msgs::Quaternion &quaternionMsg, torch::Tensor &tensor);

#endif // ROS_CONVERSIONS_H
