#ifndef EXPLORATIVE_H
#define EXPLORATIVE_H

#include "predictive.h"

class TrajectoryCost
{
public:
	TrajectoryCost () = default;

	virtual torch::Tensor value (const torch::Tensor &trajectory) = 0;

	DEF_SHARED(TrajectoryCost)
};

template<class _Cost>
class Optimizer
{
public:
	struct Params {

	};

	Optimizer () = default;

	/**
	 * @brief Get optimal trajectory according to @p _Cost
	 * @param trajectories [Nx2xM] tensor where
	 *	N: length of trajectories
	 *	M: number of trajectories to be evaluated
	 * @return index of optimal trajectory
	 */
	int getOptimal (const torch::Tensor &trajectories);
	void setParams (const Params &params);

	_Cost &cost () {
		return _cost;
	}
private:
	_Cost _cost;
	Params _params;
};

class ExplorationCost : public TrajectoryCost
{
public:
	torch::Tensor value (const torch::Tensor &trajectory) { return torch::Tensor (); }
};

using ExplorativeBase = PredictivePolicy<StraightAhead, CollisionTerminator>;

template<class _Cost>
class ExplorativePolicy : public ExplorativeBase
{
public:
	struct Params : ExplorativeBase::Params {
		nlib::Range angleRange;
	};

	ExplorativePolicy (Policy::Type type, const Napvig::Ptr &napvig):
		ExplorativeBase (type, napvig)
	{}

	Result followPolicy (const State &initialState) override;
	torch::Tensor debugHistory () override;

	DEF_SHARED(ExplorativePolicy)

protected:
	torch::Tensor generateTrajectories (const State &initialState);
	torch::Tensor _history;

	NLIB_PARAMS_INHERIT(ExplorativeBase)

private:
	Optimizer<_Cost> _optimizer;

};

class FullyExplorativePolicy : public ExplorativePolicy<ExplorationCost>
{
public:
	FullyExplorativePolicy (const Napvig::Ptr &napvig):
		ExplorativePolicy (FULLY_EXPLORATIVE, napvig)
	{}

	DEF_SHARED(FullyExplorativePolicy)
};

#endif // EXPLORATIVE_H











