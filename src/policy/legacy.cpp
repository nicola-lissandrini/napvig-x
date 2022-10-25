#include "legacy.h"

using namespace std;
using namespace torch;
using namespace lietorch;

static const torch::Tensor haltCommand = torch::tensor ({0, 0}, kFloat);

Policy::Result LegacyPolicy::followPolicy (const State &initialState) {
	boost::optional<Tensor> napvigResult = _napvig->compute (initialState);

	if (napvigResult.has_value ())
		return {*napvigResult, RESULT_ACCEPT};
	else
		return {Tensor (), RESULT_FAIL};
}

Policy::Result HaltPolicy::followPolicy (const State &initialState) {
	return {haltCommand, RESULT_ACCEPT};
}
