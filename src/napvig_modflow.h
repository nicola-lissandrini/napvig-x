#ifndef NAPVIGMODFLOW_H
#define NAPVIGMODFLOW_H

#include <memory>
#include <torch/all.h>
#include <nlib/nl_node.h>
#include <lietorch/pose.h>

#include "napvig_x.h"

class NapvigModFlow : public nlib::NlModFlow
{
public:
	NapvigModFlow();

	void loadModules () override;

	DEF_SHARED(NapvigModFlow)
	
private:
	std::vector<Policy::Ptr> _policies;
	FramesTracker::Ptr _framesTracker;
	Napvig::Ptr _napvig;
	LandmarksManager::Ptr _landmarksManager;
};

class FramesTrackerModule : public nlib::NlModule
{
public:
	FramesTrackerModule (nlib::NlModFlow *modFlow, const FramesTracker::Ptr &framesTracker);

	void initParams (const nlib::NlParams &nlParams) override;
	void setupNetwork () override;

	void measuresTriggerSlot (const torch::Tensor &);
	void odomSlot (const lietorch::Pose2 &odomFrame);
	void targetSlot (const lietorch::Pose2 &targetFrame);

private:
	FramesTracker::Ptr _framesTracker;
	nlib::Channel _measuresUpdatedChannel, _robotUpdatedChannel, _targetUpdatedChannel;
};


class NapvigXModule : public nlib::NlModule
{
public:
	NapvigXModule (nlib::NlModFlow *modFlow, const std::vector<Policy::Ptr> &policies, const FramesTracker::Ptr &framesTracker);
	
	void initParams (const nlib::NlParams &nlParams) override;
	void setupNetwork () override;
	
	void clockSlot ();
	void abortSlot ();

private:
	const std::vector<Policy::Ptr> &_policies;
	nlib::ReadyFlagsStr _flags;
	NapvigX _napvigX;
	std::map<Policy::Type, nlib::Channel> _policyChannels;
};

class LandmarksModule : public nlib::NlModule
{
public:
	LandmarksModule (nlib::NlModFlow *modFlow, const LandmarksManager::Ptr &landmarksManager, const FramesTracker::Ptr &framesTracker);

	void initParams (const nlib::NlParams &nlParams) override;
	void setupNetwork () override;

	void poseUpdated ();

private:
	FramesTracker::Ptr _framesTracker;
	LandmarksManager::Ptr _landmarksManager;
};

class PolicyModule : public nlib::NlModule
{
public:
	PolicyModule (nlib::NlModFlow *modFlow, const Policy::Ptr &policy);

	void initParams (const nlib::NlParams &nlParams) override;
	void setupNetwork () override;

	Policy::Result followPolicySlot(const State &initialState);

	void targetUpdated ();
	void measuresUpdated ();

private:
	template<class _Derived>
	typename _Derived::Ptr derived () {
		return std::dynamic_pointer_cast<_Derived> (_policy);
	}

	Policy::Ptr _policy;
	nlib::Channel _commandChannel, _napvigChannel, _historyChannel, _costDebugChannel;
};

class NapvigModule : public nlib::NlModule
{
public:
	struct Params {
		bool outputValues;
		bool outputGradients;
		nlib::Range gridRanges;
		int outputFrameSkip;
	};

	NapvigModule (nlib::NlModFlow *modFlow, const Napvig::Ptr &napvig);

	void initParams (const nlib::NlParams &nlParams) override;
	void setupNetwork () override;

	void measuresSlot (const torch::Tensor &measures);

private:
	void initDebugGrid();
	void debugValues ();
	void debugGradients ();

private:
	struct Debug {
		int frameSeq;
		torch::Tensor gridPoints;
		int gridSize;
	} _debug;

	Params _params;
	nlib::Channel _debugValuesChannel, _debugGradientsChannel, _processedMeasuresChannel;
	Napvig::Ptr _napvig;
};

class ProcessOutputs : public nlib::NlModule
{
public:
	struct Params {
		bool toWorldFrame;
	};

	enum OutputType {
		OUTPUT_MEASURES,
		OUTPUT_TENSOR_DEBUG_1,
		OUTPUT_TENSOR_DEBUG_2,
		OUTPUT_POSE_DEBUG,
		OUTPUT_VALUES,
		OUTPUT_GRADIENTS,
		OUTPUT_HISTORY,
		OUTPUT_TARGET
	};

	ProcessOutputs (nlib::NlModFlow *modFlow, const std::vector<Policy::Ptr> &policies, const FramesTracker::Ptr &framesTracker);

	void initParams (const nlib::NlParams &nlParams) override;
	void setupNetwork () override;

	void measuresSlot (const torch::Tensor &measures);
	void poseSlot (const lietorch::Pose2 &pose);
	void commandSlot (const torch::Tensor &command);

	void debugValuesSlot (const torch::Tensor &values);
	void debugGradientsSlot (const torch::Tensor &values);
	void debugHistory (const torch::Tensor &history);
	void debugCostSlot (const torch::Tensor &cost);
	void targetSlot ();

private:
	FramesTracker::Ptr _framesTracker;
	const std::vector<Policy::Ptr> &_policies;
	Params _params;
	nlib::Channel _measuresChannel, _tensorSink, _poseSink, _commandSink;
};


#endif // NAPVIGMODFLOW_H
