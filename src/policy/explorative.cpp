#include "explorative.h"
#include <lietorch/unit_complex.h>

using namespace std;
using namespace torch;
using namespace lietorch;

template
class ExplorativePolicy<ExplorationCost>;

template<class _Cost>
void Optimizer<_Cost>::setParams(const Params &params) {
	_params = params;
}

template<class _Cost>
Policy::Result ExplorativePolicy<_Cost>::followPolicy(const State &initialState)
{
	Terminator::Type cause;

	Tensor trajectories = generateTrajectories (initialState);

	_history = trajectories;

	// only debug
	return {torch::zeros ({2}, kFloat), RESULT_ACCEPT};
}

template<class _Cost>
Tensor ExplorativePolicy<_Cost>::debugHistory() {
	return _history;
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

		COUTNS(_predictHistory);
		if (cause == Terminator::TERMINATION_MAX_STEPS)
			trajectories = torch::cat ({trajectories,
										_predictHistory.unsqueeze(0)});
		// Otherwise rediction stopped prematurely, exclude directly, whatever the reason
	}

	return trajectories;
}
