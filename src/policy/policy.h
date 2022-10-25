#ifndef POLICY_H
#define POLICY_H

#include <boost/optional/optional.hpp>
#include <nlib/nl_utils.h>
#include <lietorch/pose.h>

#include "../napvig.h"

class Policy
{
public:
	enum ResultType {
		RESULT_NONE,
		RESULT_FAIL,
		RESULT_ACCEPT,
		RESULT_COMPLETE,
		RESULT_FINALIZE
	};

	struct Result {
		torch::Tensor command;
		ResultType type;
	};

	enum Type {
		IDLE = -1,
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

	virtual Result followPolicy (const State &initialState) = 0;
	virtual void updateRobot (const lietorch::Pose2 &robot) {}
	virtual void updateTarget (const lietorch::Pose2 &target) {}
	virtual torch::Tensor debugHistory () { return torch::Tensor (); }
	std::string name () const;
	Type type ();

	DEF_SHARED(Policy);

protected:
	Napvig::Ptr _napvig;

private:
	Type _type;
};








#endif // POLICY_H
