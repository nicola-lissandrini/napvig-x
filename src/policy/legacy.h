#ifndef LEGACY_H
#define LEGACY_H

#include "policy.h"

class LegacyPolicy : public Policy
{
public:
	LegacyPolicy (const Napvig::Ptr &napvig):
		  Policy (LEGACY, napvig)
	{}

	Result followPolicy (const State &initialState) override;

	DEF_SHARED(LegacyPolicy);
};

class HaltPolicy : public Policy
{
public:
	HaltPolicy (const Napvig::Ptr &napvig):
		  Policy(HALT, napvig)
	{}

	Result followPolicy (const State &initialState) override;

	DEF_SHARED(HaltPolicy);
};

#endif // LEGACY_H
