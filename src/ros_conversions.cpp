#include "ros_conversions.h"
#include "geometry_msgs/Pose.h"

using namespace torch;
using namespace torch::indexing;
using namespace lietorch;

void laserScanToTensor(const sensor_msgs::LaserScan &scanMsg, torch::Tensor &tensorMsg)
{
	const int measCount = scanMsg.ranges.size ();
	tensorMsg = torch::empty ({measCount, LASER_SCAN_DIM}, kFloat);

	for (int i = 0; i < measCount; i++) {
		float currAngle = scanMsg.angle_min + i * scanMsg.angle_increment;
		float currRadius = scanMsg.ranges[i];

		tensorMsg[i][0] = currRadius * cos (currAngle);
		tensorMsg[i][1] = currRadius * sin (currAngle);
	}
}

void pose2ToPoseMsg (const Pose2 &pose, geometry_msgs::Pose &poseMsg)
{
	poseMsg.position.x = pose.translation ().coeffs[0].item ().toFloat ();
	poseMsg.position.y = pose.translation ().coeffs[1].item ().toFloat ();

	float angle = pose.rotation ().log ().coeffs.item ().toFloat ();

	poseMsg.orientation.w = cos(angle/2);
	poseMsg.orientation.z = sin(angle/2);
}

void pointMsgToTorch (const geometry_msgs::Point &vectorMsg, Tensor &tensor) {
	tensor = torch::tensor ({vectorMsg.x,
							vectorMsg.y,
							vectorMsg.z}, torch::kFloat);
}

void quaternionMsgToTorch(const geometry_msgs::Quaternion &quaternionMsg, Tensor &tensor) {
	tensor=  torch::tensor ({quaternionMsg.x,
							quaternionMsg.y,
							quaternionMsg.z,
							quaternionMsg.w}, torch::kFloat);
}
