#include "napvig_x.h"

using namespace std;
using namespace torch;
using namespace lietorch;

static const torch::Tensor zeroState = torch::tensor ({1, 0}, kFloat);

Policy::Type NapvigX::getNext (Policy::ResultType result)
{
	switch (_currPolicy) {
	case Policy::IDLE:
		_currPolicy = Policy::FULLY_EXPLORATIVE;
		break;
	case Policy::FULLY_EXPLOITATIVE:
		switch (result) {
		case Policy::RESULT_NONE:
		case Policy::RESULT_ACCEPT:
			_currPolicy = Policy::FULLY_EXPLOITATIVE;
			break;
		case Policy::RESULT_COMPLETE:
			// Future: _currPolicy = Policy::FREE_SPACE;
			_currPolicy = Policy::HALT;
			break;
		case Policy::RESULT_FINALIZE:
		case Policy::RESULT_FAIL:
			// Future: _currPolicy = <explorative>;
			_currPolicy = Policy::HALT;
			break;
		}
	case Policy::FULLY_EXPLORATIVE:
		_currPolicy = Policy::FULLY_EXPLORATIVE;

		break;
	default:
		break;
	}

	return _currPolicy;
}

void NapvigX::reset() {
	_currPolicy = Policy::IDLE;
}

State NapvigX::getInitialization () {
	return {_currPose.translation ().coeffs, _currPose.rotation () * zeroState};
}

void NapvigX::updatePose(const lietorch::Pose2 &pose) {
	_currPose = pose;
}

