#include "policy.h"

using namespace std;
using namespace torch;
using namespace lietorch;

static const char *policyNames[] = {
	"legacy",
	"fully_exploitative",
	"fully_explorative",
	"partly_exploitative",
	"free_space",
	"halt"
};

string Policy::name () const {
	return policyNames[_type];
}

Policy::Type Policy::type() {
	return _type;
}












