#include "landscape.h"

using namespace std;
using namespace torch;
using namespace torch::indexing;

Smoother::Smoother (int dim, int samplesCount, float variance):
	  _params{dim, samplesCount, variance}
{}

Tensor Smoother::evaluate (const Fcn &f, const Tensor &x) const
{
	Tensor xVar = torch::normal (0., _params.radius,
								{_params.samplesCount, _params.dim});
	Tensor xEval = xVar + x.unsqueeze (1);

	return f(xEval).mean (1);
}

Landscape::Landscape ()
{
	_valueLambda = [this] (const Tensor &p) -> Tensor {
		return this->preSmoothValue (p);
	};

	_gradientLambda = [this] (const Tensor &p) -> Tensor {
		return this->preSmoothGradient (p);
	};
}

float Landscape::computeNoAmplificationGain () const {
	const double sqMR = pow(_params.measureRadius, 2);
	const double sqSR = pow(_params.smoothRadius, 2);
	return pow((sqMR + sqSR) / (2 * M_PI * sqMR * sqSR), 2. / float(L_DIM));
}

float Landscape::computeSmoothGain () const {
	return pow (2 * M_PI * pow(_params.smoothRadius, 2), float(L_DIM) / 2.) * computeNoAmplificationGain ();
}

Tensor Landscape::peak (const torch::Tensor &v) const {
	return (v * (-0.5 / (_params.measureRadius * _params.measureRadius))).exp ();
}

Tensor Landscape::value (const Tensor &p) const
{
	assert (_measures.size (0) > 0);

	return _smoother->evaluate (_valueLambda, p);
}

Tensor Landscape::gradient (const Tensor &p) const
{
	assert (_measures.size (0) > 0);

	if (p.sizes().size() == 1) {
		return _smoother->evaluate (_gradientLambda, p.unsqueeze(0)).squeeze ();
	} else {
		return _smoother->evaluate (_gradientLambda, p);
	}
}

bool Landscape::invalid() const {
	return _measures.size (0) == 0;
}

Tensor Landscape::preSmoothValue (const Tensor &p) const
{
	Tensor measuresDiff = p - _measures.unsqueeze (1).unsqueeze (2);
	Tensor distToMeasures = measuresDiff.pow (2).sum (3);

	auto [collapsedDist, idxes] = distToMeasures.min(0);

	return peak (collapsedDist) * _smoothGain;
}

Tensor Landscape::preSmoothGradient (const Tensor &p) const
{
	Tensor measuresDiff = p - _measures.unsqueeze (1).unsqueeze (2);
	Tensor distToMeasures = measuresDiff.pow (2).sum (3);

	auto [collapsedDist, idxes] = distToMeasures.min(0);

	Tensor collapsedDiff = measuresDiff.permute ({1,2,0,3})
							   .index({_xGrid.slice(1, 0, idxes.numel ()),
									   _yGrid.slice (1, 0, idxes.numel ()),
									   idxes.reshape ({1, -1}),
									   Ellipsis})
							   .reshape ({-1, _params.precision, L_DIM});

	return collapsedDiff;// / (2 * _params.measureRadius * _params.measureRadius) * peak (collapsedDist).unsqueeze(2) * _smoothGain;
}

Tensor Landscape::setMeasures (const Tensor &measures) {
	Tensor decimated = measures.slice (0, 0, torch::nullopt, _params.decimation);
	Tensor indexes = decimated.isfinite ().sum(1).nonzero ().squeeze ();

	_measures = decimated.index ({indexes});

	return _measures;
}

void Landscape::setParams (const Params &params) {
	_smoother = make_shared<Smoother> (L_DIM, params.precision, params.smoothRadius);

	_params = params;

	// Init indexing grid
	auto xyGrid = torch::meshgrid ({torch::arange (0, _params.batchSize),
								   torch::arange (0, _params.precision)});

	_xGrid = xyGrid[0].reshape({1,-1});
	_yGrid = xyGrid[1].reshape({1,-1});

	_smoothGain = computeSmoothGain ();
}
