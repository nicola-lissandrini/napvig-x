#include "frames_tracker.h"

using namespace std;
using namespace torch;
using namespace lietorch;

FramesTracker::FramesTracker()
{
	_frameSet.addFlag (FRAME_ROBOT);
	_frameSet.addFlag (FRAME_MEASURES);
	_frameSet.addFlag (FRAME_TARGET);

	_frames[FRAME_WORLD] = lietorch::Pose2 ();
}

template
	void FramesTracker::update<FRAME_ROBOT> (const lietorch::Pose2 &worldPose);
template
	void FramesTracker::update<FRAME_TARGET> (const lietorch::Pose2 &worldPose);

template<Frame frame>
void FramesTracker::update (const lietorch::Pose2 &worldPose) {
	_frameSet.set (frame);
	_frames[frame] = worldPose;
}

void FramesTracker::updateMeasures () {
	if (!_frameSet[FRAME_ROBOT])
		return;

	_frameSet.set (FRAME_MEASURES);
	_frames[FRAME_MEASURES] = _frames[FRAME_ROBOT];
}

Pose2 FramesTracker::getTf (Frame toFrame, Frame fromFrame) {
	assert (_frameSet[fromFrame] && _frameSet[toFrame]);

	return _frames[fromFrame].inverse() * _frames[toFrame];
}

Tensor FramesTracker::getIn (Frame toFrame, Frame fromFrame, const Tensor &point) {
	assert (_frameSet[fromFrame] && _frameSet[toFrame]);

	return _frames[toFrame].inverse () * _frames[fromFrame] * point;
}

Tensor FramesTracker::getIn (const Pose2 &toFrame, Frame fromFrame, const Tensor &point) {
	assert (_frameSet[fromFrame]);

	return toFrame.inverse () * _frames[fromFrame] * point;
}

Tensor FramesTracker::getIn (Frame toFrame, const Pose2 &fromFrame, const Tensor &point) {
	assert (_frameSet[toFrame]);

	return _frames[toFrame].inverse () * fromFrame * point;
}

Pose2 FramesTracker::get (Frame frame) {
	return _frames[frame];
}

bool FramesTracker::ready(Frame frame) {
	return _frameSet[frame];
}


