#ifndef NAPVIGX_H
#define NAPVIGX_H

#include "policy/policy.h"
#include "policy/legacy.h"
#include "policy/exploitative.h"
#include "policy/explorative.h"

class NapvigX
{
public:
	NapvigX () {}

	void reset ();
	Policy::Type getNext (Policy::ResultType result);
	State getInitialization ();
	void updatePose (const lietorch::Pose2 &pose);

	DEF_SHARED(NapvigX);

private:
	lietorch::Pose2 _currPose;
	Policy::Type _currPolicy;
};

#endif // NAPVIGX_H
