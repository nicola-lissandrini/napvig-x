#include "legacy.h"

using namespace std;
using namespace torch;
using namespace lietorch;

static const torch::Tensor haltCommand = torch::tensor ({0, 0}, kFloat);

Policy::Result LegacyPolicy::followPolicy (const State &initialState) {
	boost::optional<Tensor> napvigResult = _napvig->compute (initialState);

	//_history = torch::cat ({napvigResult->unsqueeze(0), initialState.search.unsqueeze(0)}, 1);

	if (napvigResult.has_value ())
		return {*napvigResult, RESULT_ACCEPT};
	else
		return {Tensor (), RESULT_FAIL};
}

Tensor LegacyPolicy::debugHistory() {
	return _history;
}

Policy::Result HaltPolicy::followPolicy (const State &initialState) {
	return {haltCommand, RESULT_ACCEPT};
}

Policy::Result FreeSpacePolicy::followPolicy (const State &initialState)
{
	if (!_framesTracker->ready (FRAME_TARGET))
		return {Tensor (), RESULT_FAIL};

	targetUpdated ();

	if (_targetInMeasures.norm ().item().toFloat () < _params.reachThreshold)
		return {Tensor (), RESULT_COMPLETE};

	return {_targetInMeasures, RESULT_ACCEPT};
}

void FreeSpacePolicy::setParams (const Params &params) {
	_params = params;
}

void FreeSpacePolicy::targetUpdated() {
	if (!_framesTracker->ready (FRAME_MEASURES))
		return;

	_targetInMeasures = _framesTracker->getTf (FRAME_TARGET, FRAME_MEASURES).translation().coeffs;
}
