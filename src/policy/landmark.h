#ifndef LANDMARK_H
#define LANDMARK_H

#include <chrono>
#include <deque>
#include <torch/all.h>
#include <lietorch/pose.h>

using Clock = std::chrono::steady_clock;
using Time = std::chrono::time_point<Clock>;

class Landmark
{
public:
	Landmark (const lietorch::Pose2 &frame):
		_creationFrame(frame),
		_creationTime(Clock::now())
	{}

	float elapsed() const;
	torch::Tensor get (const lietorch::Pose2 &currentFrame);

private:
	lietorch::Pose2 _creationFrame;
	Time _creationTime;
};

using LandmarksBatch = std::deque<Landmark>;

#endif // LANDMARK_H
