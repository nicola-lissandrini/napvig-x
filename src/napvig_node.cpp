#include "napvig_node.h"
#include "nav_msgs/Odometry.h"
#include "ros/forwards.h"
#include "ros_conversions.h"
#include <lietorch/pose.h>
#include <lietorch/quaternion.h>
#include <lietorch/rn.h>
#include <nlib/nl_ros_conversions.h>
#include <stdexcept>
#include "std_msgs/Float32MultiArray.h"
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose2D.h>

using namespace std;
using namespace nlib;
using namespace torch;
using namespace lietorch;

NapvigNode::NapvigNode(int &argc, char **argv, const std::string &name, uint32_t options):
	  Base(argc, argv, name, options)
{
	init<ModFlow> ();

	_measuresChannel = sources()->declareSource<Tensor> ("measures_source");
	_odomChannel = sources()->declareSource<Pose2> ("odom_source");
	_clockChannel = sources()->declareSource<> ("clock_source");

	sinks()->declareSink ("publish_tensor", &NapvigNode::publishTensor, this);
	sinks()->declareSink ("publish_pose", &NapvigNode::publishPose, this);
	sinks()->declareSink ("publish_command", &NapvigNode::publishCommand, this);

	finalizeModFlow ();
}

void NapvigNode::initParams () {}

vector<const char *> outputStrings = {
	"measures",
	"tensor_debug_1",
	"tensor_debug_2",
	"pose_debug",
	"values",
	"gradients"
};

void NapvigNode::initROS ()
{
	addSub ("measures", _nlParams.get<int> ("topics/queue_size", 1), &NapvigNode::measuresCallback);
	addSub ("odom", _nlParams.get<int> ("topics/queue_size", 1), &NapvigNode::odomCallback);
	addPub<geometry_msgs::Pose2D> ("command", _nlParams.get<int> ("topics/queue_size", 1));

	string prefix = _nlParams.get<string> ("topics/output_prefix");

	vector<ProcessOutputs::OutputType> outputTypes = _nlParams.get<ProcessOutputs::OutputType, vector> ("topics/outputs", outputStrings);
	for (auto currType : outputTypes) {
		string topicName = prefix + "/" + outputStrings[currType];

		switch (currType) {
		case ProcessOutputs::OUTPUT_POSE_DEBUG:
			addPub<geometry_msgs::PoseStamped> (outputStrings[currType], topicName, 1);
			break;
		default:
			addPub<std_msgs::Float32MultiArray> (outputStrings[currType], topicName, 1);
			break;
		}

	}
}

void NapvigNode::measuresCallback (const sensor_msgs::LaserScan &scanMsg)
{
	Tensor measures;

	laserScanToTensor (scanMsg, measures);

	sources()->callSource (_measuresChannel, measures);
}

void NapvigNode::odomCallback (const nav_msgs::Odometry &odomMsg)
{
	Tensor position, orientation;

	pointMsgToTorch (odomMsg.pose.pose.position, position);
	quaternionMsgToTorch (odomMsg.pose.pose.orientation, orientation);

	lietorch::Quaternion quaternion(orientation);

	lietorch::Pose2 pose2(position.slice(0,0,2), UnitComplex (quaternion.log ().coeffs[2].item().toFloat ()));
	sources()->callSource (_odomChannel, pose2);
}

void NapvigNode::publishTensor (const torch::Tensor &tensor, ProcessOutputs::OutputType outputType)
{
	std_msgs::Float32MultiArray tensorMsg;

	tensorToMsg (tensor, {static_cast<float> (outputType)}, tensorMsg);

	try {
		publish (outputStrings[outputType], tensorMsg);
	} catch (const std::out_of_range &e) {} // ignore non declared outputs
}

void NapvigNode::publishPose (const lietorch::Pose2 &pose, ProcessOutputs::OutputType outputType)
{
	geometry_msgs::PoseStamped poseMsg;

	pose2ToPoseMsg (pose, poseMsg.pose);
	poseMsg.header.frame_id = "map";

	try {
		publish (outputStrings[outputType], poseMsg);
	} catch (const std::out_of_range &) {} // ignore non-declared outputs
}

void NapvigNode::publishCommand (const torch::Tensor &command) {
	geometry_msgs::Pose2D commandMsg;

	commandMsg.x = command[0].item ().toFloat ();
	commandMsg.y = command[1].item ().toFloat ();

	publish ("command", commandMsg);
}

void NapvigNode::onSynchronousClock (const ros::TimerEvent &timeEvent)
{
	sources()->callSource (_clockChannel);
}

int main (int argc, char *argv[])
{
	NapvigNode nn(argc, argv, "napvig");

	return nn.spin ();
}
