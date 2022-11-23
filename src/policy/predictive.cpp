#include "predictive.h"
#include <nlib/nl_utils.h>

using namespace std;
using namespace torch;
using namespace lietorch;

const char *terminationStrings[] = {
	"none",
	"collision",
	"max_steps",
	"target_approached",
	"fault"
};

template
	class PredictivePolicy<TowardsTarget, CollisionTargetTerminator>;

template
	class PredictivePolicy<StraightAhead, CollisionTerminator>;

void Terminator::reset() {
	_stepsCount = 0;
	_cause = TERMINATION_NONE;
}

bool Terminator::check (const at::Tensor &) {
	if (_stepsCount == params().maxCount) {
		_cause = TERMINATION_MAX_STEPS;
		return true;
	} else {
		_stepsCount++;
		return false;
	}
}

Terminator::Type Terminator::cause() const {
	return _cause;
}

void CollisionTerminator::reset () {
	Terminator::reset ();
}

bool CollisionTerminator::check (const Tensor &sample)
{
	if (Terminator::check (sample))
		return true;

	if (_napvig->collides (sample)) {
		_cause = TERMINATION_COLLISION;
		return true;
	}

	return false;
}

bool CollisionTargetTerminator::check (const Tensor &sample)
{
	if (CollisionTerminator::check (sample))
		return true;

	if ((sample - _target).norm().item().toFloat() < params().targetRadius) {
		_cause = TERMINATION_TARGET_APPROACHED;
		return true;
	}

	return false;
}

void CollisionTargetTerminator::updateTarget (const Tensor &target) {
	_target = target;
}

void TowardsTarget::reset (const State &initialState) {
}

void TowardsTarget::updateTarget(const at::Tensor &target) {
	_target = target;
}


torch::Tensor normalizedDiff (const Tensor &a, const Tensor &b) {
	Tensor diff = b - a;

	return diff / diff.norm();
}

Tensor TowardsTarget::getSearch (const Tensor &trajectory) {
	return normalizedDiff (trajectory[-1], _target);
}

void StraightAhead::reset (const State &initialState) {
	_first = initialState.search;
}

torch::Tensor StraightAhead::getSearch (const torch::Tensor &trajectory) {
	if (trajectory.size(0) == 1)
		return _first;
	else
		return normalizedDiff(trajectory[-2], trajectory[-1]);
}

template<class _Predictor, class _Terminator>
Tensor PredictivePolicy<_Predictor, _Terminator>::predict (const State &initialState, Terminator::Type &terminationCause)
{
	Tensor trajectory = initialState.position.unsqueeze (0);
	Tensor searchHistory = torch::empty ({0, 2}, kFloat);
	std::optional<Tensor> next = initialState.position;

	_predictor.reset (initialState);
	_terminator.reset ();


	while (!_terminator.check (*next)) {
		// Get napvig initialization according to predictor policy
		Tensor search = _predictor.getSearch (trajectory);

		searchHistory = torch::cat ({searchHistory,
									 search.unsqueeze(0)});

		// Compute next step
		next = _napvig->compute (State {*next, search});


		// If napvig failes, abort prediction
		if (!next.has_value ()) {
			terminationCause = Terminator::TERMINATION_FAULT;
			return trajectory;
		}

		// Store sample in trajectory
		trajectory = torch::cat ({trajectory,
								 next->unsqueeze(0)});
	}

	Tensor lastSearch = _predictor.getSearch (trajectory);
	searchHistory = torch::cat ({searchHistory,
								 lastSearch.unsqueeze(0)});

	_predictHistory = torch::cat ({trajectory, searchHistory}, 1);
	terminationCause = _terminator.cause ();


	return trajectory;
}
