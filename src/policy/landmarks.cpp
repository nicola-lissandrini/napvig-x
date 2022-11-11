#include "landmarks.h"

using namespace std;
using namespace torch;
using namespace lietorch;

static const torch::Tensor origin = torch::zeros ({2}, kFloat);

Landmark::Landmark (const Pose2 &pose):
	Landmark (pose, origin, 1.f)
{}

Landmark::Landmark (const Pose2 &pose, const torch::Tensor &position, float weight):
	  _creationPose(Clock::now (), pose),
	  _position(position),
	  _weight(weight)
{}

torch::Tensor Landmark::get (const Pose2 &currentPose) const
{
	Pose2 oldToCurrent = currentPose.inverse () * _creationPose.obj ();

	return oldToCurrent * _position;
}

float Landmark::elapsed () const {
	chrono::duration<float> elapsed = Clock::now () - _creationPose.time ();

	return elapsed.count ();
}

void LandmarksBatch::add (const Landmark &landmark) {
	_landmarks.push_back (landmark);

	if (_landmarks.size () > _params.size) {
		_landmarks.pop_front ();
	}
}

const Landmark &LandmarksBatch::last () const {
	return _landmarks.back ();
}

void LandmarksBatch::setParams (const Params &params) {
	_params = params;
}

bool LandmarksBatch::isEmpty() const {
	return _landmarks.size () == 0;
}

LandmarksBatch::Queue::const_iterator LandmarksBatch::end() const { return _landmarks.cend (); }
LandmarksBatch::Queue::const_iterator LandmarksBatch::begin() const { return _landmarks.cbegin (); }

void LandmarksManager::setParams(const Params &params) {
	_params = params;
	_landmarks.setParams (LandmarksBatch::Params {_params.batchSize});
}

void LandmarksManager::update (const Pose2 &currentPose)
{
	if (_landmarks.isEmpty ()) {
		_landmarks.add (Landmark (currentPose));
		return;
	}

	const Landmark &last = _landmarks.last ();

	if (last.get (currentPose).norm ().item().toFloat () > _params.minDistance)
		_landmarks.add (Landmark (currentPose));
	if (last.elapsed () > _params.minElapsed)
		_landmarks.add (Landmark (currentPose));

}

void LandmarksManager::addInvalidExplored (const lietorch::Pose2 &frame, const Tensor &trajectory)
{
	for (int i = 0; i < trajectory.size(0); i++)
		_landmarks.add(Landmark (frame, trajectory[i], _params.invalidWeight));
}
