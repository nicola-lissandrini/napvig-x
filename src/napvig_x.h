#ifndef NAPVIGX_H
#define NAPVIGX_H

#include <nlib/nl_utils.h>
#include <lietorch/pose.h>

#include "napvig.h"

class Policy
{
public:
	enum Type {
		LEGACY,
		FULLY_EXPLOITATIVE,
		FULLY_EXPLORATIVE,
		PARTLY_EXPLOITATIVE,
		FREE_SPACE,
		HALT
	};

	Policy (Type type, const Napvig::Ptr &napvig):
		  _type(type),
		  _napvig(napvig)
	{}

	virtual boost::optional<torch::Tensor> followPolicy (const State &initialState) = 0;
	std::string name () const;
	Type type ();

	DEF_SHARED(Policy);

protected:
	Napvig::Ptr _napvig;

private:
	Type _type;
};

class LegacyPolicy : public Policy
{
public:
	LegacyPolicy (const Napvig::Ptr &napvig):
		  Policy (LEGACY, napvig)
	{}

	boost::optional<torch::Tensor> followPolicy (const State &initialState) override;

	DEF_SHARED(LegacyPolicy);
};

class HaltPolicy : public Policy
{
public:
	HaltPolicy (const Napvig::Ptr &napvig):
		  Policy(HALT, napvig)
	{}

	boost::optional<torch::Tensor> followPolicy (const State &initialState) override;

	DEF_SHARED(HaltPolicy)
};

class NapvigX
{
public:
	NapvigX () {}

	Policy::Type getNext ();
	Policy::Type getFirst ();
	State getInitial ();
	void updatePose (const lietorch::Pose2 &pose);

	DEF_SHARED(NapvigX);
private:
	lietorch::Pose2 _currPose;
	Policy::Type _currPolicy;
};

#endif // NAPVIGX_H
