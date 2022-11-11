#ifndef LEGACY_H
#define LEGACY_H

#include "policy.h"

class LegacyPolicy : public Policy
{
public:
	LegacyPolicy (const Napvig::Ptr &napvig, const FramesTracker::Ptr &framesTracker, const LandmarksManager::Ptr &landmarksManager):
		  Policy (LEGACY, napvig, framesTracker, landmarksManager)
	{}

	Result followPolicy (const State &initialState) override;
	torch::Tensor debugHistory () override;

	DEF_SHARED(LegacyPolicy);

private:
	torch::Tensor _history;
};

class HaltPolicy : public Policy
{
public:
	HaltPolicy (const Napvig::Ptr &napvig, const FramesTracker::Ptr &framesTracker, const LandmarksManager::Ptr &landmarksManager):
		  Policy(HALT, napvig, framesTracker, landmarksManager)
	{}

	Result followPolicy (const State &initialState) override;

	DEF_SHARED(HaltPolicy);
};

class FreeSpacePolicy : public Policy
{
public:
	struct Params {
		float reachThreshold;
	};

	FreeSpacePolicy (const Napvig::Ptr &napvig, const FramesTracker::Ptr &framesTracker, const LandmarksManager::Ptr &landmarksManager):
		  Policy (FREE_SPACE, napvig, framesTracker, landmarksManager)
	{}

	void setParams (const Params &params);
	void targetUpdated () override;
	Result followPolicy (const State &initialState) override;

	DEF_SHARED(FreeSpacePolicy);

private:
	torch::Tensor _targetInMeasures;
	Params _params;
};



#endif // LEGACY_H
