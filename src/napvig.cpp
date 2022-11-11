#include "napvig.h"
#include <lietorch/unit_complex.h>

using namespace torch;
using namespace std;

static const lietorch::UnitComplex piHalf(M_PI_2);

Tensor Napvig::setMeasures (const Tensor &measures) {
	return _landscape.setMeasures (measures);
}

void Napvig::setParams (const Params &napvigParams, const Landscape::Params &landscapeParams)
{
	_params = napvigParams;
	_landscape.setParams (landscapeParams);
}

Tensor Napvig::debugLandscapeValues (const Tensor &grid) const {
	Tensor values = _landscape.value (grid);

	return torch::cat ({grid, values.unsqueeze(1)}, 1);
}

Tensor Napvig::debugLandscapeGradients (const Tensor &grid) const {
	Tensor gradients = _landscape.gradient (grid);

	return torch::cat ({grid, gradients}, 1);
}

Tensor Napvig::projectOnto (const Tensor &space, const Tensor &vector) const {
	return space.inner (vector) * space;
}

Tensor Napvig::getOrthogonal (const Tensor &r) const {
	Tensor rotated = piHalf * r;

	return rotated / rotated.norm ();
}

Tensor Napvig::voronoiSearch (const Tensor &xStep, const Tensor &r) const
{
	Tensor searchSpace = getOrthogonal (r);
	Tensor xCurr = xStep;
	int iterCount;
	bool terminationCondition = false;

	for (iterCount = 0; !terminationCondition; iterCount++) {
		// Get gradient
		Tensor gradientCurr = _landscape.gradient (xCurr);

		// Project onto the search space
		Tensor gradientProjected = projectOnto (searchSpace, gradientCurr);

		// Update rule
		Tensor xNext = xCurr + _params.gradientStepSize * gradientProjected;

		// Compute progress
		float updateDistance = (xNext - xCurr).norm ().item ().toFloat ();

		// Update current point
		xCurr = xNext;

		// Check termination
		terminationCondition = (updateDistance < _params.terminationDistance) ||
							   (iterCount >= _params.maxIterations);
	}

	return xCurr;
}

Tensor Napvig::stepAhead (const State &x0) const {
	return x0.position + _params.stepAheadSize * x0.search;
}

boost::optional<Tensor> Napvig::compute (const State &initialization) const {
	Tensor xStep;

	if (!_landscape.isInitialized ())
		return boost::none;

	xStep = stepAhead (initialization);

	if (_landscape.isEmpty ())
		return xStep;

	return voronoiSearch (xStep, initialization.search);
}

bool Napvig::collides (const Tensor &sample)
{
	// if there are no obstacles distToObstacles returns NAN
	// any comparison with NAN is false
	return _landscape.distToObstacles (sample) < _params.collisionRadius;
}
























