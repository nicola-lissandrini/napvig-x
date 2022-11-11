#ifndef LANDMARKS_H
#define LANDMARKS_H

#include <nlib/nl_utils.h>
#include <lietorch/pose.h>
#include <deque>

using Clock = std::chrono::steady_clock;
template<class T>
using Timed = nlib::TimedObject<T, Clock, Clock::duration>;

class Landmark
{
public:
	Landmark (const lietorch::Pose2 &pose);
	Landmark (const lietorch::Pose2 &pose,
			  const torch::Tensor &position,
			  float weight);

	Landmark (const Landmark &) = default;
	Landmark &operator = (const Landmark &) = default;

	torch::Tensor get (const lietorch::Pose2 &currentPose) const;
	float elapsed () const;

	DEF_SHARED(Landmark)

private:
	Timed<lietorch::Pose2> _creationPose;
	torch::Tensor _position;
	float _weight;
};

class LandmarksBatch
{
	using Queue = std::deque<Landmark>;

public:
	struct Params {
		int size;
	};

	LandmarksBatch () = default;

	void add (const Landmark &landmark);
	void setParams (const Params &params);
	bool isEmpty () const;
	const Landmark &last () const;

	Queue::const_iterator begin () const;
	Queue::const_iterator end () const;

	DEF_SHARED(LandmarksBatch)

private:
	Queue _landmarks;
	Params _params;
};

class LandmarksManager
{
public:
	struct Params {
		int batchSize;
		float minElapsed;
		float minDistance;
		float invalidWeight;
	};

	LandmarksManager () = default;

	void setParams (const Params &params);
	void update (const lietorch::Pose2 &currentPose);
	void addInvalidExplored (const lietorch::Pose2 &frame, const torch::Tensor &trajectory);
	const LandmarksBatch &landmarks () const { return _landmarks; }

	DEF_SHARED(LandmarksManager)

private:
	Params _params;
	LandmarksBatch _landmarks;
};


#endif // LANDMARKS_H
