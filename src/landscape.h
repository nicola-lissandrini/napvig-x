#ifndef LANDSCAPE_H
#define LANDSCAPE_H

#include <torch/all.h>
#include <nlib/nl_utils.h>

class Smoother
{
	struct Params {
		int dim;
		int samplesCount;
		float radius;
	};

public:
	using Fcn = std::function<torch::Tensor(const torch::Tensor &)>;

	Smoother (int dim, int samplesCount, float radius);

	torch::Tensor evaluate (const Fcn &f, const torch::Tensor &x) const;

	DEF_SHARED(Smoother);

protected:
	Params _params;
};

#define L_DIM 2

class Landscape
{
public:
	struct Params {
		float measureRadius;
		float smoothRadius;
		int precision;
		int batchSize; // number of simultaneous landscape points in evaluation
		int decimation;
	};

	Landscape ();

	torch::Tensor value (const torch::Tensor &p) const;
	torch::Tensor gradient (const torch::Tensor &p) const;
	float distToObstacles(const torch::Tensor &p) const;

	bool invalid () const;
	torch::Tensor setMeasures (const torch::Tensor &measures);
	void setParams (const Params &params);

	DEF_SHARED(Landscape);

private:
	torch::Tensor peak (const torch::Tensor &v) const;
	torch::Tensor preSmoothValue (const torch::Tensor &p) const;
	torch::Tensor preSmoothGradient (const torch::Tensor &p) const;

	float computeNoAmplificationGain () const;
	float computeSmoothGain () const;

private:
	Smoother::Fcn _valueLambda;
	Smoother::Fcn _gradientLambda;
	Smoother::Ptr _smoother;
	torch::Tensor _xGrid, _yGrid;
	torch::Tensor _measures;
	Params _params;

	float _smoothGain;
};

#endif // LANDSCAPE_H
