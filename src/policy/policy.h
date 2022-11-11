#ifndef POLICY_H
#define POLICY_H

#include <boost/optional/optional.hpp>
#include <nlib/nl_utils.h>
#include <lietorch/pose.h>

#include "../napvig.h"
#include "../frames_tracker.h"
#include "landmarks.h"

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
		PARTLY_EXPLORATIVE,
		FREE_SPACE,
		HALT
	};

	Policy (Type type,
		   const Napvig::Ptr &napvig,
		   const FramesTracker::Ptr &framesTracker,
		   const LandmarksManager::Ptr &landmarksManager):
		  _type(type),
		  _napvig(napvig),
		  _framesTracker(framesTracker),
		  _landmarksManager(landmarksManager)
	{}

	virtual Result followPolicy (const State &initialState) = 0;
	virtual void measuresUpdated () {}
	virtual void targetUpdated () {}

	virtual torch::Tensor debugHistory () { return torch::Tensor (); }
	virtual torch::Tensor debugCost () { return torch::Tensor (); }
	std::string name () const;
	Type type ();

	DEF_SHARED(Policy);

protected:
	Napvig::Ptr _napvig;
	LandmarksManager::Ptr _landmarksManager;
	FramesTracker::Ptr _framesTracker;

private:
	Type _type;
};

inline void debugGrid (const nlib::Range &range, torch::Tensor &gridPoints, int &gridSize)
{
	torch::Tensor xyRange = torch::arange (range.min, range.max, *range.step, torch::dtype (torch::kFloat));
	torch::Tensor xx, yy;
	std::vector<torch::Tensor> xy;

	xy = torch::meshgrid ({xyRange, xyRange});

	xx = xy[0].reshape (-1);
	yy = xy[1].reshape (-1);

	gridPoints = torch::stack ({xx, yy}, 1);
	gridSize = xyRange.size (0);
}







#endif // POLICY_H
