#include "napvig_modflow.h"
#include <cmath>
#include <lietorch/pose.h>

using namespace std;
using namespace torch;
using namespace torch::indexing;
using namespace nlib;
using namespace lietorch;

NapvigModFlow::NapvigModFlow():
	  nlib::NlModFlow ()
{
	_napvig = make_shared<Napvig> ();

	_policies = {
		make_shared<LegacyPolicy> (_napvig),
		make_shared<HaltPolicy> (_napvig),
		make_shared<FullyExploitativePolicy> (_napvig),
		make_shared<FullyExplorativePolicy> (_napvig)
	};
}

void NapvigModFlow::loadModules ()
{
	loadModule<FramesTrackerModule> ();
	loadModule<NapvigXModule> (_policies);
	loadModule<NapvigModule> (_napvig);
	for (Policy::Ptr policy : _policies)
		loadModule<PolicyModule> (policy);
	loadModule<ProcessOutputs> (_policies);
}

FramesTrackerModule::FramesTrackerModule(nlib::NlModFlow *modFlow):
	  nlib::NlModule (modFlow, "frames_tracker")
{
	_flags.addFlag ("measures_set");
	_flags.addFlag ("odom_set");
	_flags.addFlag ("target_set");
}

NapvigXModule::NapvigXModule(nlib::NlModFlow *modFlow, const std::vector<Policy::Ptr> &policies):
	  NlModule(modFlow, "napvig_x"),
	  _policies(policies)
{
	_flags.addFlag ("pose_set");
}

PolicyModule::PolicyModule(nlib::NlModFlow *modFlow, const Policy::Ptr &policy):
	  NlModule (modFlow, policy->name () + "_policy"),
	  _policy(policy)
{
}

NapvigModule::NapvigModule(nlib::NlModFlow *modFlow, const Napvig::Ptr &napvig):
	  nlib::NlModule (modFlow, "napvig"),
	  _napvig(napvig)
{}

ProcessOutputs::ProcessOutputs(nlib::NlModFlow *modFlow, const std::vector<Policy::Ptr> &policies):
	  nlib::NlModule (modFlow, "outputs"),
	  _policies(policies)
{}


/***********************
 * Network set-up
 * *********************/

void FramesTrackerModule::setupNetwork ()
{
	requestConnection ("measures_source", &FramesTrackerModule::measuresTriggerSlot);
	requestConnection ("odom_source", &FramesTrackerModule::odomSlot);
	requestConnection ("target_source", &FramesTrackerModule::targetSlot);

	_toMeasuresFrameChannel = createChannel<Pose2> ("to_measures_frame");
	_toRobotFrameChannel = createChannel<Pose2> ("to_robot_frame");
	_targetChannel = createChannel<Pose2> ("target_frame");
}

void PolicyModule::setupNetwork()
{
	requestConnection ("follow_" + _policy->name (), &PolicyModule::followPolicySlot);
	requestConnection ("target_frame", &PolicyModule::updateTarget);

	_commandChannel = createChannel<Tensor> ("command_" + _policy->name ());
	_historyChannel = createChannel<Tensor> ("history_" + _policy->name ());
}

void NapvigXModule::setupNetwork ()
{
	requestConnection ("clock_source", &NapvigXModule::clockSlot);
	requestConnection ("to_measures_frame", &NapvigXModule::poseSlot);
	requestEnablingChannel ("measures_source");

	// _policyChannels.resize (_policies.size ());

	for (const Policy::Ptr &policy : _policies)
		_policyChannels[policy->type()] = createChannel<State> ("follow_" + policy->name ());
}

void NapvigModule::setupNetwork ()
{
	requestConnection ("measures_source", &NapvigModule::measuresSlot);

	_debugValuesChannel = createChannel<Tensor> ("debug_values");
	_debugGradientsChannel = createChannel<Tensor> ("debug_gradients");
	_processedMeasuresChannel = createChannel<Tensor> ("measures_processed");
}

void ProcessOutputs::setupNetwork ()
{
	requestConnection ("measures_processed", &ProcessOutputs::measuresSlot);
	requestConnection ("to_robot_frame", &ProcessOutputs::poseSlot);
	requestConnection ("debug_values", &ProcessOutputs::debugValuesSlot);
	requestConnection ("debug_gradients", &ProcessOutputs::debugGradientsSlot);
	requestConnection ("target_frame", &ProcessOutputs::targetSlot);

	for (const Policy::Ptr &policy : _policies) {
		requestConnection ("command_" + policy->name (), &ProcessOutputs::commandSlot);
		requestConnection ("history_" + policy->name (), &ProcessOutputs::debugHistory);
	}

	_tensorSink = requireSink<Tensor, OutputType> ("publish_tensor");
	_poseSink = requireSink<lietorch::Pose2, OutputType> ("publish_pose");
	_commandSink = requireSink<Tensor> ("publish_command");
}


/***********************
 * Parameters
 * *********************/

void FramesTrackerModule::initParams (const NlParams &nlParams) {}
void NapvigXModule::initParams (const NlParams &nlParams) {}

void PolicyModule::initParams(const nlib::NlParams &nlParams)
{
	switch (_policy->type ()) {
	case Policy::HALT:
	case Policy::LEGACY:
		break;
	case Policy::FULLY_EXPLOITATIVE: {
		FullyExploitativeBase::Params params;

		params.terminator.maxCount = nlParams.get<int> ("max_count");
		params.terminator.collisionRadius  = nlParams.get<float> ("collision_radius");
		params.terminator.targetRadius = nlParams.get<float> ("target_radius");

		derived<FullyExploitativePolicy> ()->setParams (params);
		break;
	}
	case Policy::FULLY_EXPLORATIVE: {
		FullyExplorativePolicy::Params params;

		params.terminator.maxCount = nlParams.get<int> ("max_count");
		params.terminator.collisionRadius  = nlParams.get<float> ("collision_radius");
		params.angleRange = nlParams.get<nlib::Range> ("angle_search_range");

		derived<FullyExplorativePolicy> ()->setParams (params);
	}
	default:
		break;
	}
}

void NapvigModule::initParams (const NlParams &nlParams) {
	_params = {
		.outputValues = nlParams.get<bool> ("debug/output_values"),
		.outputGradients = nlParams.get<bool> ("debug/output_gradient"),
		.gridRanges = nlParams.get<nlib::Range> ("debug/output_range"),
		.outputFrameSkip = nlParams.get<int> ("debug/frames_skip")
	};

	initDebugGrid ();

	Landscape::Params landscapeParams = {
		.measureRadius = nlParams.get<float> ("landscape/measure_radius"),
		.smoothRadius = nlParams.get<float> ("landscape/smooth_radius"),
		.precision = nlParams.get<int> ("landscape/precision"),
		.batchSize = static_cast<int> (_debug.gridPoints.size (0)),
		.decimation = nlParams.get<int> ("landscape/decimation")
	};

	Napvig::Params napvigParams;

	napvigParams = {
		.stepAheadSize = nlParams.get<float> ("step_ahead_size"),
		.gradientStepSize = nlParams.get<float> ("gradient_step_size"),
		.terminationDistance = nlParams.get<float> ("termination_distance"),
		.maxIterations = nlParams.get<int> ("max_iterations")
	};

	_napvig->setParams (napvigParams, landscapeParams);

	_debug.frameSeq = 0;
}

void NapvigModule::initDebugGrid ()
{
	Tensor xyRange = torch::arange (_params.gridRanges.min, _params.gridRanges.max, *_params.gridRanges.step, torch::dtype (kFloat));
	Tensor xx, yy, testGrid;
	vector<Tensor> xy;

	xy = meshgrid ({xyRange, xyRange});

	xx = xy[0].reshape (-1);
	yy = xy[1].reshape (-1);

	_debug.gridPoints = torch::stack ({xx, yy}, 1);
	_debug.gridSize = xyRange.size (0);
}

void ProcessOutputs::initParams (const NlParams &nlParams) {
	_params = {
		.toWorldFrame = nlParams.get<bool> ("to_world_frame", false)
	};
}

/***********************
 * Slots
 * *********************/

void FramesTrackerModule::odomSlot (const lietorch::Pose2 &odomFrame)
{
	_flags.set ("odom_set");
	_lastFrame = odomFrame;
	
	if (_flags.all ()) {
		_odomToRobotFrame = odomFrame.inverse () * _measuresFrame;
		_robotToMeasuresFrame = _odomToRobotFrame.inverse ();

		emit (_toRobotFrameChannel, _odomToRobotFrame);
		emit (_toMeasuresFrameChannel, _robotToMeasuresFrame);
	}
}

void FramesTrackerModule::targetSlot(const lietorch::Pose2 &targetFrame)
{
	if (!_flags["odom_set"] || !_flags["measures_set"])
		return;

	_flags.set("target_set");

	lietorch::Pose2 targetInMeasures = _measuresFrame.inverse () * targetFrame;

	emit (_targetChannel, targetInMeasures);
}

void FramesTrackerModule::measuresTriggerSlot (const torch::Tensor &)
{
	if (!_flags["odom_set"])
		return;

	_flags.set ("measures_set");
	_measuresFrame = _lastFrame;
}

void NapvigXModule::poseSlot (const Pose2 &pose) {
	_flags.set ("pose_set");
	_napvigX.updatePose (pose);
}

void NapvigXModule::clockSlot ()
{
	if (!_flags.all ())
		return;

	_napvigX.reset ();

	Policy::Type policy;
	Policy::ResultType result = Policy::RESULT_NONE;
	
	while (result != Policy::RESULT_ACCEPT) {
		policy = _napvigX.getNext (result);
		result = callService<Policy::ResultType> (_policyChannels[policy], _napvigX.getInitialization ());
	}
}

Policy::ResultType PolicyModule::followPolicySlot(const State &initialState)
{
	Policy::Result result = _policy->followPolicy (initialState);

	Tensor debugHistory = _policy->debugHistory ();

	if (debugHistory.numel () > 0)
		emit (_historyChannel, debugHistory);

	if (result.type == Policy::RESULT_ACCEPT)
		emit (_commandChannel, result.command);

	return result.type;
}

void PolicyModule::updateTarget (const Pose2 &target) {
	_policy->updateTarget (target);
}



void NapvigModule::debugGradients () {
	Tensor gradients = _napvig->debugLandscapeGradients (_debug.gridPoints);

	emit (_debugGradientsChannel, gradients);
}

void NapvigModule::debugValues ()
{
	double taken;
	Tensor values;
	PROFILE_N (taken, [&] {
			values = _napvig->debugLandscapeValues (_debug.gridPoints);
		}, values.size(0));

	emit(_debugValuesChannel, values);
}

void NapvigModule::measuresSlot (const Tensor &measures)
{
	Tensor processedMeasures = _napvig->setMeasures (measures);

	emit (_processedMeasuresChannel, processedMeasures);

	if (_debug.frameSeq == _params.outputFrameSkip) {
		_debug.frameSeq = 0;

		if (_params.outputValues)
			debugValues ();
		if (_params.outputGradients)
			debugGradients ();
	} else
		_debug.frameSeq++;
}

void ProcessOutputs::commandSlot (const Tensor &command)
{
	if (command.norm().item ().toFloat () < 1e-7){
		emit (_commandSink, command);
		return;
	}

	Tensor commandInRobotFrame = _toRobotFrame * command;

	emit (_commandSink, commandInRobotFrame);
}

void ProcessOutputs::debugValuesSlot (const at::Tensor &values) {
	emit (_tensorSink, values, OUTPUT_VALUES);
}

void ProcessOutputs::debugGradientsSlot (const at::Tensor &values) {
	emit (_tensorSink, values, OUTPUT_GRADIENTS);
}

void ProcessOutputs::debugHistory (const Tensor &history) {
	emit (_tensorSink, history, OUTPUT_HISTORY);
}

void ProcessOutputs::targetSlot (const lietorch::Pose2 &target) {
	emit (_tensorSink, target.translation ().coeffs, OUTPUT_TARGET);
}

void ProcessOutputs::measuresSlot (const torch::Tensor &measures)
{
	if (_params.toWorldFrame) {
		// not implemented
	}

	emit (_tensorSink, measures, OUTPUT_MEASURES);
}

void ProcessOutputs::poseSlot (const lietorch::Pose2 &frame) {
	_toRobotFrame = frame;
	emit (_poseSink, frame, OUTPUT_POSE_DEBUG);
}





















