#ifndef NAPVIGX_H
#define NAPVIGX_H

#include "policy/policy.h"
#include "policy/legacy.h"
#include "policy/exploitative.h"
#include "policy/explorative.h"
#include "frames_tracker.h"

class NapvigX
{
public:
	struct Params {
		int legacyHoldCount;
	};

	NapvigX (const FramesTracker::Ptr &framesTracker);

	void reset ();
	Policy::Type getNext (Policy::ResultType result);
	State getInitialization ();
	void setParams (const Params &params);
	bool ready ();
	void finalize (const torch::Tensor &result);

	DEF_SHARED(NapvigX);

private:
	bool holdLegacy ();

private:
	Params _params;
	FramesTracker::Ptr _framesTracker;
	torch::Tensor _lastSearch;
	lietorch::Pose2 _lastMeasuresPose;
	Policy::Type _currPolicy;
	int _holdCount;
};

#endif // NAPVIGX_H
