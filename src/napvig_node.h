#ifndef NAPVIGNODE_H
#define NAPVIGNODE_H

#include "napvig_modflow.h"

#include <lietorch/pose.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32MultiArray.h>


class NapvigNode : public nlib::NlNode<NapvigNode>
{
	NL_NODE(NapvigNode)

	using ModFlow = NapvigModFlow;

public:
	NapvigNode (int &argc, char **argv, const std::string &name, uint32_t options = 0);

	void initROS ();
	void initParams ();

	void measuresCallback (const sensor_msgs::LaserScan &scanMsg);
	void odomCallback (const nav_msgs::Odometry &odomMsg);
	void targetCallback (const geometry_msgs::Pose &targetMsg);

	void publishTensor (const torch::Tensor &tensor, ProcessOutputs::OutputType outputType);
	void publishPose (const lietorch::Pose2 &pose, ProcessOutputs::OutputType);
	void publishCommand (const torch::Tensor &command);

	void abort ();

	DEF_SHARED(NapvigNode)

protected:
	void onSynchronousClock (const ros::TimerEvent &timeEvent);

private:
	nlib::Channel _measuresChannel;
	nlib::Channel _odomChannel;
	nlib::Channel _targetChannel;
	nlib::Channel _clockChannel;
	nlib::Channel _abortChannel;
};

#endif // NAPVIGNODE_H
