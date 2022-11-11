#include "explorative.h"
#include <lietorch/unit_complex.h>

using namespace std;
using namespace torch;
using namespace torch::indexing;
using namespace lietorch;

const torch::Tensor zeroPoint = torch::zeros ({2}, kFloat);

template
class ExplorativePolicy<ExplorationCost>;

template
class ExplorativePolicy<TargetExplorationCost>;

void TrajectoryCost::updatePose (const lietorch::Pose2 &currentPose) {
	_currentPose = currentPose;
}

Tensor ExplorationCost::values (const torch::Tensor &trajectories)
{
	Tensor values = torch::zeros ({trajectories.size(0)});

	for (const Landmark &landmark : _landmarksManager->landmarks ()) {
		// Get landmark in current measures frame
		Tensor landmarkPos = landmark.get (_currentPose);

		// Get differences from landmark to trajectories samples
		Tensor trajectoryDiff = landmarkPos - trajectories.index ({Slice(), Slice(), Slice(0,2)});

		// Get decay factor e^(-lambda (t - t_l)
		float decayFactor = exp (-params().decayConstant * landmark.elapsed ());

		// Get cost associated to current landmark:
		// J_l = w e^(-lambda (t - t_l) e^(-(x_t - x_l)^2/radius)
		values += params().weight * decayFactor * (-trajectoryDiff.norm (2,2).pow(2) / (2 * pow(params().landmarkRadius,2))).exp().sum(1);
	}

	return values;
}

template<class _Cost>
void ExplorativePolicy<_Cost>::setParams(const Params &params)
{
	ExplorativeBase::setParams (params);

	_cost.setParams (params.cost);
}

template<class _Cost>
void ExplorativePolicy<_Cost>::measuresUpdated () {
	_cost.updatePose (_framesTracker->get (FRAME_MEASURES));
}

template<class _Cost>
Tensor ExplorativePolicy<_Cost>::getOptimal (const torch::Tensor &trajectories)
{
	Tensor values = _cost.values (trajectories);

	return trajectories[values.argmin (0)];
}

template<class _Cost>
Policy::Result ExplorativePolicy<_Cost>::followPolicy (const State &initialState)
{
	Tensor trajectories = generateTrajectories (initialState);

	if (trajectories.size(0) == 0)
		return {Tensor (), RESULT_FAIL};

	Tensor trajectory = getOptimal (trajectories);

	_history = torch::cat ({trajectories, _cost.values (trajectories).unsqueeze(1).unsqueeze(2).expand({-1,trajectories.size(1),-1})}, 2);

	return {trajectory.index ({1, Slice(0,2)}), RESULT_ACCEPT};
}

Policy::Result PartlyExplorativePolicy::followPolicy (const State &initialState)
{
	if (!_framesTracker->ready(FRAME_TARGET))
		return {Tensor (), RESULT_FAIL};

	// Local target pose might have changed wrt measures frame
	targetUpdated();

	return ExplorativePolicy<TargetExplorationCost>::followPolicy (initialState);
}

template<class _Cost>
Tensor ExplorativePolicy<_Cost>::debugCost()
{
	if (!params().outputCost)
		return Tensor ();

	Tensor values = _cost.values (_debug.gridPoints.unsqueeze (1));

	return torch::cat ({_debug.gridPoints, values.unsqueeze (1)}, 1);
}

template<class _Cost>
Tensor ExplorativePolicy<_Cost>::debugHistory() {
	return _history;
}

template<class _Cost>
void ExplorativePolicy<_Cost>::initDebugGrid() {
	debugGrid (params().outputRange, _debug.gridPoints, _debug.gridSize);
}

template<class _Cost>
Tensor ExplorativePolicy<_Cost>::generateTrajectories (const State &initialState)
{
	const int maxCount =  params().terminator.maxCount + 1;
	Tensor trajectories = torch::empty ({0, maxCount , 4}, kFloat);
	Tensor currTrajectory;
	State currInitial = initialState;
	Terminator::Type cause;

	float angle = 0;

	for (int i = 0; angle < params().angleRange.max; i++) {
		angle = i * params().angleRange.step.value() + params().angleRange.min;

		currInitial.search = UnitComplex(angle) * initialState.search;
		currTrajectory = predict (currInitial, cause);

		cout << "Prediction terminated due to \e[34m" << terminationStrings[cause] << "\e[0m" << endl;

		if (cause == Terminator::TERMINATION_MAX_STEPS)
			trajectories = torch::cat ({trajectories,
										_predictHistory.unsqueeze(0)});
		else // Otherwise rediction stopped prematurely, exclude directly, whatever the reason
			if (cause != Terminator::TERMINATION_TARGET_APPROACHED)
				// Add landmarks in explored invalid trajectories
				_landmarksManager->addInvalidExplored(_framesTracker->get(FRAME_MEASURES),
													  currTrajectory);
	}
	return trajectories;
}

Tensor TargetExplorationCost::values (const at::Tensor &trajectories)
{
	Tensor explorationValues = ExplorationCost::values (trajectories);

	if (_targetInMeasures.numel () == 0)
		return explorationValues;

	Tensor distToTarget = (trajectories.index ({Slice(), Slice(), Slice(0,2)})
						   - _targetInMeasures.unsqueeze(0).unsqueeze(1))
						   .norm(2, 2).pow(2).sum(1);


	return explorationValues + params().targetWeight * distToTarget;
}

void TargetExplorationCost::updateTarget (const Tensor &targetPos) {
	_targetInMeasures = targetPos;
}

void PartlyExplorativePolicy::targetUpdated ()
{
	if (!_framesTracker->ready (FRAME_MEASURES))
		return;

	_cost.updateTarget (_framesTracker->getTf (FRAME_TARGET, FRAME_MEASURES).translation ().coeffs);
}

























