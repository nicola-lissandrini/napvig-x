#include "napvig_x.h"

using namespace std;
using namespace torch;
using namespace lietorch;

static const torch::Tensor zeroState = torch::tensor ({1, 0}, kFloat);
static const torch::Tensor haltCommand = torch::tensor ({0, 0}, kFloat);

Policy::Type NapvigX::getNext ()
{
	switch (_currPolicy) {
	default:
		_currPolicy = Policy::HALT;
		break;
	}

	return _currPolicy;
}

Policy::Type NapvigX::getFirst() {
	_currPolicy = Policy::LEGACY;
	return _currPolicy;
}

State NapvigX::getInitial () {
	return {_currPose.translation ().coeffs, _currPose.rotation () * zeroState};
}

void NapvigX::updatePose(const lietorch::Pose2 &pose) {
	_currPose = pose;
}

static const char *policyNames[] = {
	"legacy",
	"fully_exploitative",
	"fully_explorative",
	"partly_exploitative",
	"free_space",
	"halt"
};

string Policy::name () const {
	return policyNames[_type];
}

Policy::Type Policy::type() {
	return _type;
}

boost::optional<Tensor> LegacyPolicy::followPolicy (const State &initialState) {
	return _napvig->compute (initialState);
}

boost::optional<Tensor> HaltPolicy::followPolicy (const State &initialState) {
	return haltCommand;
}
