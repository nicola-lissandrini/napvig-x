#ifndef NAPVIG_H
#define NAPVIG_H

#include "landscape.h"

struct State {
	torch::Tensor position;
	torch::Tensor search;

	DEF_SHARED(State);
};

class Napvig
{
public:
	struct Params {
		float stepAheadSize;
		float gradientStepSize;
		float terminationDistance;
		int maxIterations;
	};

	Napvig () = default;

	void setParams (const Params &napvigParams,
					const Landscape::Params &landscapeParams);
	torch::Tensor setMeasures(const torch::Tensor &measures);

	boost::optional<torch::Tensor> compute (const State &initialization) const;

	torch::Tensor debugLandscapeValues (const torch::Tensor &grid) const;
	torch::Tensor debugLandscapeGradients (const torch::Tensor &grid) const;

	DEF_SHARED(Napvig);

private:
	torch::Tensor stepAhead (const State &x0) const;
	torch::Tensor voronoiSearch (const torch::Tensor &xStep, const torch::Tensor &r) const;
	torch::Tensor projectOnto (const torch::Tensor &space, const torch::Tensor &vector) const;
	torch::Tensor getOrthogonal (const torch::Tensor &r) const;

private:
	Landscape _landscape;
	Params _params;
};

#endif // NAPVIG_H
