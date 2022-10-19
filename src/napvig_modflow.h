#ifndef NAPVIGMODFLOW_H
#define NAPVIGMODFLOW_H

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
	Napvig::Ptr _napvig;
};

class FramesTrackerModule : public nlib::NlModule
{
public:
	FramesTrackerModule (nlib::NlModFlow *modFlow);

	void initParams (const nlib::NlParams &nlParams) override;
	void setupNetwork () override;

	void measuresTriggerSlot (const torch::Tensor &);
	void odomSlot (const lietorch::Pose2 &odomFrame);

private:
	nlib::ReadyFlagsStr _flags;
	lietorch::Pose2 _measuresFrame, _lastFrame;
	nlib::Channel _toMeasuresFrameChannel, _toRobotFrameChannel;
};


class NapvigXModule : public nlib::NlModule
{
public:
	NapvigXModule (nlib::NlModFlow *modFlow, const std::vector<Policy::Ptr> &policies);
	
	void initParams (const nlib::NlParams &nlParams) override;
	void setupNetwork () override;
	
	void clockSlot ();
	void poseSlot (const lietorch::Pose2 &);
	
private:
	const std::vector<Policy::Ptr> &_policies;
	nlib::ReadyFlagsStr _flags;
	NapvigX _napvigX;
	std::map<Policy::Type, nlib::Channel> _policyChannels;
};

class PolicyModule : public nlib::NlModule
{
public:
	PolicyModule (nlib::NlModFlow *modFlow, const Policy::Ptr &policy);

	void initParams (const nlib::NlParams &nlParams) override;
	void setupNetwork () override;

	bool followPolicySlot (const State &initialState);

private:
	Policy::Ptr _policy;
	nlib::Channel _commandChannel, _napvigChannel;
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
		OUTPUT_GRADIENTS
	};

	ProcessOutputs (nlib::NlModFlow *modFlow, const std::vector<Policy::Ptr> &policies);

	void initParams (const nlib::NlParams &nlParams) override;
	void setupNetwork () override;

	void measuresSlot (const torch::Tensor &measures);
	void poseSlot (const lietorch::Pose2 &pose);
	void commandSlot (const torch::Tensor &command);
	void debugValuesSlot (const torch::Tensor &values);
	void debugGradientsSlot (const torch::Tensor &values);

private:
	lietorch::Pose2 _toRobotFrame;
	const std::vector<Policy::Ptr> &_policies;
	Params _params;
	nlib::Channel _measuresChannel, _tensorSink, _poseSink, _commandSink;
};


#endif // NAPVIGMODFLOW_H
