#ifndef EXPLORATIVE_H
#define EXPLORATIVE_H

#include "predictive.h"
#include "landmarks.h"
#include <deque>

class TrajectoryCost
{
public:
	TrajectoryCost () = default;

	virtual torch::Tensor values (const torch::Tensor &trajectories) = 0;
	void updatePose (const lietorch::Pose2 &currentPose);

	DEF_SHARED(TrajectoryCost)

protected:
	lietorch::Pose2 _currentPose;
};

class ExplorationCost : public TrajectoryCost
{
public:
	struct Params {
		float landmarkRadius;
		float weight;
		float decayConstant;

		virtual ~Params () {}

		DEF_SHARED(Params)
	};

	ExplorationCost (const LandmarksManager::Ptr &landmarksManager):
		  _landmarksManager(landmarksManager)
	{}

	torch::Tensor values (const torch::Tensor &trajectories) override;

	NLIB_PARAMS_SET

protected:
	NLIB_PARAMS_BASE

private:
	NLIB_PARAMS_INHERIT(ExplorationCost)

	LandmarksManager::Ptr _landmarksManager;
	Params::Ptr _params;
};

class TargetExplorationCost : public ExplorationCost
{
public:
	struct Params : ExplorationCost::Params {
		float targetWeight;

		DEF_SHARED(Params)
	};

	TargetExplorationCost (const LandmarksManager::Ptr &landmarksManager):
		ExplorationCost (landmarksManager)
	{}

	torch::Tensor values (const torch::Tensor &trajectories) override;
	void updateTarget (const torch::Tensor &targetPos);

private:
	NLIB_PARAMS_INHERIT(ExplorationCost)

	torch::Tensor _targetInMeasures;
};

using ExplorativeBase = PredictivePolicy<StraightAhead, CollisionTerminator>;

template<class _Cost>
class ExplorativePolicy : public ExplorativeBase
{
public:
	struct Params : ExplorativeBase::Params {
		nlib::Range angleRange;
		typename _Cost::Params cost;
		bool outputCost;

		nlib::Range outputRange;

		DEF_SHARED(Params)
	};

	ExplorativePolicy (Policy::Type type, const Napvig::Ptr &napvig, const FramesTracker::Ptr &framesTracker, const LandmarksManager::Ptr &landmarksManager):
		  ExplorativeBase (type, napvig, framesTracker, landmarksManager),
		  _cost(landmarksManager)
	{}

	void setParams (const Params &params);
	void measuresUpdated () override;
	virtual Result followPolicy (const State &initialState) override;
	torch::Tensor debugCost () override;
	torch::Tensor debugHistory () override;
	void initDebugGrid ();

	DEF_SHARED(ExplorativePolicy)

protected:
	/**
	 * @brief Get optimal trajectory according to @p TrajectoryCost
	 * @param trajectories [NxMx2] tensor where
	 *	N: length of trajectories
	 *	M: number of trajectories to be evaluated
	 * @return index of optimal trajectory
	 */
	torch::Tensor getOptimal(const torch::Tensor &trajectories);
	torch::Tensor generateTrajectories (const State &initialState);
	torch::Tensor _history;

	NLIB_PARAMS_INHERIT(ExplorativeBase)

	struct Debug {
		torch::Tensor gridPoints;
		int gridSize;
	} _debug;
	_Cost _cost;
	lietorch::Pose2 _currentPose;
};

class FullyExplorativePolicy : public ExplorativePolicy<ExplorationCost>
{
public:
	FullyExplorativePolicy (const Napvig::Ptr &napvig, const FramesTracker::Ptr &framesTracker, const LandmarksManager::Ptr &landmarksManager):
		ExplorativePolicy (FULLY_EXPLORATIVE, napvig, framesTracker, landmarksManager)
	{}

	DEF_SHARED(FullyExplorativePolicy)
};

class PartlyExplorativePolicy : public ExplorativePolicy<TargetExplorationCost>
{
public:
	PartlyExplorativePolicy (const Napvig::Ptr &napvig, const FramesTracker::Ptr &framesTracker, const LandmarksManager::Ptr &landmarksManager):
		ExplorativePolicy (PARTLY_EXPLORATIVE, napvig, framesTracker, landmarksManager)
	{}
	
	void targetUpdated () override;
	Result followPolicy (const State &initialState) override;

	DEF_SHARED(PartlyExplorativePolicy)
};

#endif // EXPLORATIVE_H











