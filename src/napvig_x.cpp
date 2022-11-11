#include "napvig_x.h"

using namespace std;
using namespace torch;
using namespace lietorch;

static const torch::Tensor zeroState = torch::tensor ({1, 0}, kFloat);

NapvigX::NapvigX(const FramesTracker::Ptr &framesTracker):
	  _holdCount(0),
	  _currPolicy(Policy::IDLE),
	  _lastSearch(zeroState),
	  _framesTracker(framesTracker)
{}

Policy::Type NapvigX::getNext (Policy::ResultType result)
{
	// Common actions for these result types
	switch (result) {
	case Policy::RESULT_COMPLETE:
		_currPolicy = Policy::HALT;
		return _currPolicy;
		break;
	case Policy::RESULT_FINALIZE:
		_currPolicy = Policy::FREE_SPACE;
		return _currPolicy;
		break;
	default:
		break;
	}

	switch (_currPolicy) {
	case Policy::IDLE:
		_currPolicy = Policy::FULLY_EXPLOITATIVE;
		break;
	case Policy::FULLY_EXPLOITATIVE:
		if (result == Policy::RESULT_FAIL)
			// Future: _currPolicy = <explorative>;
			_currPolicy = Policy::PARTLY_EXPLORATIVE;
		else // result == Policy::RESULT_ACCEPT
			_currPolicy = Policy::LEGACY;
		break;

	case Policy::PARTLY_EXPLORATIVE:
	case Policy::FULLY_EXPLORATIVE:
		if (result == Policy::RESULT_FAIL)
			// No escape: every trajectory collides
			_currPolicy = Policy::HALT;
		else // result == Policy::RESULT_ACCEPT
			_currPolicy = Policy::LEGACY;
		break;

	case Policy::LEGACY:
		if (holdLegacy ())
			_currPolicy = Policy::LEGACY;
		else
			_currPolicy = Policy::FULLY_EXPLOITATIVE;
		break;
	case Policy::HALT:
		_currPolicy = Policy::FULLY_EXPLOITATIVE;
		break;
	default:
		break;
	}

	return _currPolicy;
}

void NapvigX::finalize (const Tensor &result) {
	if (_currPolicy != Policy::LEGACY) {
		if (result.norm ().item().toFloat () < 1e-7)
			_lastSearch = zeroState;
		else
			_lastSearch = result / result.norm ();
		_lastMeasuresPose = _framesTracker->get (FRAME_MEASURES);
	}
}

bool NapvigX::holdLegacy ()
{
	if (_holdCount == _params.legacyHoldCount) {
		_holdCount = 0;
		_lastSearch = zeroState;
		return false;
	} else {
		_holdCount++;
		return true;
	}
}

void NapvigX::reset() {
}

State NapvigX::getInitialization () {
	Pose2 robotInMeasures = _framesTracker->getTf (FRAME_MEASURES, FRAME_ROBOT);
	Tensor lastSearchInCurrent = _framesTracker->getIn (FRAME_MEASURES, _lastMeasuresPose, _lastSearch);

	return {robotInMeasures.translation ().coeffs, robotInMeasures.rotation () * lastSearchInCurrent};
}

void NapvigX::setParams (const Params &params) {
	_params = params;
}

bool NapvigX::ready () {
	return _framesTracker->ready (FRAME_MEASURES) && _framesTracker->ready (FRAME_ROBOT);
}


