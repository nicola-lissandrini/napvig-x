#ifndef PREDICTIVE_H
#define PREDICTIVE_H

#include "policy.h"
#include <nlib/nl_utils.h>
#include <memory>

class Terminator
{
public:
	struct Params {
		int maxCount;

		// Enable inheritance
		virtual ~Params () {}
		DEF_SHARED(Params);
	};

	enum Type {
		TERMINATION_NONE,
		TERMINATION_COLLISION,
		TERMINATION_MAX_STEPS,
		TERMINATION_TARGET_APPROACHED,
		TERMINATION_FAULT
	};

	Terminator ():
		  _cause(TERMINATION_NONE),
		  _stepsCount(0)
	{}

	virtual void reset ();
	virtual bool check (const torch::Tensor &sample);
	Type cause () const;

	NLIB_PARAMS_SET;

protected:
	Type _cause;

	NLIB_PARAMS_BASE;

private:
	NLIB_PARAMS_INHERIT(Terminator)

private:
	int _stepsCount;
	Params::Ptr _params;
};

class CollisionTerminator : public Terminator
{
public:
	// TODO: all inherited params are stored for each subclass, fix that
	struct Params : Terminator::Params {
		float collisionRadius;

		DEF_SHARED(Params)
	};

	CollisionTerminator (const Napvig::Ptr &napvig):
		  _napvig(napvig)
	{}

	virtual void reset () override;
	virtual bool check (const torch::Tensor &sample) override;

protected:
	const Napvig::Ptr &_napvig;

private:
	NLIB_PARAMS_INHERIT(Terminator)
};

class CollisionTargetTerminator : public CollisionTerminator
{
public:
	struct Params : CollisionTerminator::Params {
		float targetRadius;
	};

	CollisionTargetTerminator (const Napvig::Ptr &napvig):
		  CollisionTerminator(napvig)
	{}

	virtual bool check (const torch::Tensor &sample) override;

	void updateTarget (const torch::Tensor &target);


protected:
	torch::Tensor _target;

private:
	NLIB_PARAMS_INHERIT(Terminator)
};

class Predictor
{
public:
	struct Params {
		virtual ~Params () {}

		DEF_SHARED(Params)
	};

	Predictor () = default;

	virtual void reset (const State &initialState) = 0;
	virtual torch::Tensor getSearch (const torch::Tensor &trajectory) = 0;

	NLIB_PARAMS_SET;

protected:
	NLIB_PARAMS_BASE;

private:
	Params::Ptr _params;

};

class TowardsTarget : public Predictor
{
public:
	struct Params : Predictor::Params {

	};

	TowardsTarget () = default;

	void reset (const State &initialState) override;
	torch::Tensor getSearch (const torch::Tensor &trajectory) override;

	void updateTarget (const torch::Tensor &target);

protected:
	torch::Tensor _target;

private:
	NLIB_PARAMS_INHERIT(Predictor);
};

class StraightAhead : public Predictor
{
public:
	struct Params : Predictor::Params {
	};

	StraightAhead () = default;

	void reset (const State &initialState) override;
	torch::Tensor getSearch (const torch::Tensor &trajectory) override;

private:
	torch::Tensor _first;

private:
	NLIB_PARAMS_INHERIT(Predictor)
};

template<class _Predictor, class _Terminator>
class PredictivePolicy : public Policy
{
public:
	struct Params {
		typename _Terminator::Params terminator;
		typename _Predictor::Params predictor;

		virtual ~Params () {}
		DEF_SHARED(Params)
	};

	PredictivePolicy (Type type, const Napvig::Ptr &napvig):
		  Policy(type, napvig),
		  _terminator(napvig)
	{}

	template<class _DerivedParams>
	void setParams (const _DerivedParams &params)
	{
		_params = std::make_shared<_DerivedParams> (params);
		_predictor.setParams (params.predictor);
		_terminator.setParams (params.terminator);
	}

protected:
	torch::Tensor predict (const State &initialState, Terminator::Type &terminationCause);

protected:
	torch::Tensor _predictHistory;
	_Predictor _predictor;
	_Terminator _terminator;

	NLIB_PARAMS_BASE;

private:
	typename Params::Ptr _params;
};

#endif // PREDICTIVE_H
