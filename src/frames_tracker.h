#ifndef FRAMESTRACKER_H
#define FRAMESTRACKER_H

#include <nlib/nl_utils.h>
#include <lietorch/pose.h>
#include <map>

#define FRAMES_COUNT 4

enum Frame {
	FRAME_WORLD = 0,
	FRAME_ROBOT,
	FRAME_MEASURES,
	FRAME_TARGET
};

class FramesTracker
{
public:
	FramesTracker ();

	template<Frame frame>
	void update (const lietorch::Pose2 &worldPose);
	void updateMeasures ();
	
	lietorch::Pose2 getTf (Frame toFrame, Frame fromFrame);
	torch::Tensor getIn (const lietorch::Pose2 &toFrame, Frame fromFrame, const torch::Tensor &point);
	torch::Tensor getIn (Frame toFrame, const lietorch::Pose2 &fromFrame, const torch::Tensor &point);
	torch::Tensor getIn (Frame toFrame, Frame fromFrame, const torch::Tensor &point);
	lietorch::Pose2 get (Frame frame);

	bool ready (Frame frame);

	DEF_SHARED(FramesTracker)

private:
	nlib::ReadyFlags<Frame> _frameSet;
	std::array<lietorch::Pose2, FRAMES_COUNT> _frames;
};

#endif // FRAMESTRACKER_H
