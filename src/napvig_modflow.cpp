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
	_landmarksManager = make_shared<LandmarksManager> ();
	_framesTracker = make_shared<FramesTracker> ();

	_policies = {
		make_shared<LegacyPolicy> (_napvig, _framesTracker, _landmarksManager),
		make_shared<FullyExploitativePolicy> (_napvig, _framesTracker, _landmarksManager),
		make_shared<FullyExplorativePolicy> (_napvig, _framesTracker, _landmarksManager),
		make_shared<PartlyExplorativePolicy> (_napvig, _framesTracker, _landmarksManager),
		make_shared<FreeSpacePolicy> (_napvig, _framesTracker, _landmarksManager),
		make_shared<HaltPolicy> (_napvig, _framesTracker, _landmarksManager)
	};
}

void NapvigModFlow::loadModules ()
{
	loadModule<FramesTrackerModule> (_framesTracker);
	loadModule<NapvigModule> (_napvig);
	loadModule<NapvigXModule> (_policies, _framesTracker);
	loadModule<LandmarksModule> (_landmarksManager, _framesTracker);
	for (Policy::Ptr policy : _policies)
		loadModule<PolicyModule> (policy);
	loadModule<ProcessOutputs> (_policies, _framesTracker);
}

FramesTrackerModule::FramesTrackerModule(nlib::NlModFlow *modFlow, const FramesTracker::Ptr &framesTracker):
	  nlib::NlModule (modFlow, "frames_tracker"),
	  _framesTracker(framesTracker)
{
}

LandmarksModule::LandmarksModule(nlib::NlModFlow *modFlow, const LandmarksManager::Ptr &landmarksManager, const FramesTracker::Ptr &framesTracker):
	  nlib::NlModule (modFlow, "landmarks"),
	  _landmarksManager(landmarksManager),
	  _framesTracker(framesTracker)
{}

NapvigXModule::NapvigXModule(nlib::NlModFlow *modFlow, const std::vector<Policy::Ptr> &policies, const FramesTracker::Ptr &framesTracker):
	  NlModule(modFlow, "napvig_x"),
	  _policies(policies),
	  _napvigX(framesTracker)
{
	_flags.addFlag ("pose_set");
}

PolicyModule::PolicyModule(nlib::NlModFlow *modFlow, const Policy::Ptr &policy):
	  NlModule (modFlow, policy->name () + "_policy"),
	  _policy(policy)
{}

NapvigModule::NapvigModule(nlib::NlModFlow *modFlow, const Napvig::Ptr &napvig):
	  nlib::NlModule (modFlow, "napvig"),
	  _napvig(napvig)
{}

ProcessOutputs::ProcessOutputs(nlib::NlModFlow *modFlow, const std::vector<Policy::Ptr> &policies, const FramesTracker::Ptr &framesTracker):
	  nlib::NlModule (modFlow, "outputs"),
	  _policies(policies),
	  _framesTracker(framesTracker)
{}


/***********************
 * Network set-up
 * *********************/

void FramesTrackerModule::setupNetwork ()
{
	requestConnection ("measures_source", &FramesTrackerModule::measuresTriggerSlot);
	requestConnection ("odom_source", &FramesTrackerModule::odomSlot);
	requestConnection ("target_source", &FramesTrackerModule::targetSlot);

	_measuresUpdatedChannel = createChannel<> ("measures_updated");
	_robotUpdatedChannel = createChannel<> ("robot_updated");
	_targetUpdatedChannel = createChannel<> ("target_updated");
}


void NapvigXModule::setupNetwork ()
{
	requestConnection ("clock_source", &NapvigXModule::clockSlot);
	requestConnection ("abort_source", &NapvigXModule::abortSlot);

	for (const Policy::Ptr &policy : _policies)
		_policyChannels[policy->type()] = createChannel<State> ("follow_" + policy->name ());
}

void PolicyModule::setupNetwork()
{
	requestConnection ("follow_" + _policy->name (), &PolicyModule::followPolicySlot);
	requestConnection ("target_updated", &PolicyModule::targetUpdated);
	requestConnection ("measures_updated", &PolicyModule::measuresUpdated);

	_commandChannel = createChannel<Tensor> ("command_" + _policy->name ());
	_historyChannel = createChannel<Tensor> ("history_" + _policy->name ());
	_costDebugChannel = createChannel<Tensor> ("cost_" + _policy->name ());
}

void LandmarksModule::setupNetwork () {
	requestConnection ("measures_updated", &LandmarksModule::poseUpdated);
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
	requestConnection ("debug_values", &ProcessOutputs::debugValuesSlot);
	requestConnection ("debug_gradients", &ProcessOutputs::debugGradientsSlot);
	requestConnection ("cost_fully_explorative", &ProcessOutputs::debugCostSlot);
	requestConnection ("cost_partly_explorative", &ProcessOutputs::debugCostSlot);
	requestConnection ("target_updated", &ProcessOutputs::targetSlot);

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
void NapvigXModule::initParams (const NlParams &nlParams)
{
	NapvigX::Params params = {
		.legacyHoldCount = nlParams.get<int> ("legacy_hold_count")
	};

	_napvigX.setParams (params);
}

void LandmarksModule::initParams (const NlParams &nlParams)
{
	LandmarksManager::Params params = {
		.batchSize = nlParams.get<int> ("batch_size"),
		.minElapsed = nlParams.get<float> ("min_elapsed"),
		.minDistance = nlParams.get<float> ("min_distance"),
		.invalidWeight = nlParams.get<float> ("invalid_weight")
	};

	_landmarksManager->setParams (params);
}

void PolicyModule::initParams(const nlib::NlParams &nlParams)
{
	switch (_policy->type ()) {
	case Policy::HALT:
	case Policy::LEGACY:
		break;
	case Policy::FREE_SPACE: {
		FreeSpacePolicy::Params params;

		params.reachThreshold = nlParams.get<float> ("reach_threshold");

		derived<FreeSpacePolicy> ()->setParams (params);

		break;
	}
	case Policy::FULLY_EXPLOITATIVE: {
		FullyExploitativeBase::Params params;

		params.terminator.maxCount = nlParams.get<int> ("max_count");
		params.terminator.targetRadius = nlParams.get<float> ("target_radius");

		derived<FullyExploitativePolicy> ()->setParams (params);
		break;
	}
	case Policy::PARTLY_EXPLORATIVE: {
		PartlyExplorativePolicy::Params params;

		params.terminator.maxCount = nlParams.get<int> ("max_count");
		params.angleRange = nlParams.get<nlib::Range> ("angle_search_range");
		params.cost.landmarkRadius = nlParams.get<float> ("cost/landmark_radius");
		params.cost.weight = nlParams.get<float> ("cost/weight");
		params.cost.targetWeight = nlParams.get<float> ("cost/target_weight");
		params.cost.decayConstant = nlParams.get<float> ("cost/decay_constant");
		params.outputCost = nlParams.get<bool> ("debug/output_cost");
		params.outputRange = nlParams.get<nlib::Range> ("debug/output_range");

		derived<PartlyExplorativePolicy> ()->setParams (params);
		derived<PartlyExplorativePolicy> ()->initDebugGrid ();

		break;
	}
	case Policy::FULLY_EXPLORATIVE: {
		FullyExplorativePolicy::Params params;

		params.terminator.maxCount = nlParams.get<int> ("max_count");
		params.angleRange = nlParams.get<nlib::Range> ("angle_search_range");
		params.cost.landmarkRadius = nlParams.get<float> ("cost/landmark_radius");
		params.cost.weight = nlParams.get<float> ("cost/weight");
		params.cost.decayConstant = nlParams.get<float> ("cost/decay_constant");
		params.outputCost = nlParams.get<bool> ("debug/output_cost");
		params.outputRange = nlParams.get<nlib::Range> ("debug/output_range");

		derived<FullyExplorativePolicy> ()->setParams (params);
		derived<FullyExplorativePolicy> ()->initDebugGrid ();

		break;
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
		.collisionRadius = nlParams.get<float> ("collision_radius"),
		.maxIterations = nlParams.get<int> ("max_iterations")
	};

	_napvig->setParams (napvigParams, landscapeParams);

	_debug.frameSeq = 0;
}


void NapvigModule::initDebugGrid () {
	debugGrid (_params.gridRanges, _debug.gridPoints, _debug.gridSize);
}

void ProcessOutputs::initParams (const NlParams &nlParams) {
	_params = {
		.toWorldFrame = nlParams.get<bool> ("to_world_frame", false)
	};
}

/***********************
 * Slots
 * *********************/

void FramesTrackerModule::odomSlot (const lietorch::Pose2 &odomFrame) {
	_framesTracker->update<FRAME_ROBOT> (odomFrame);
	emit (_robotUpdatedChannel);
}

void FramesTrackerModule::targetSlot (const lietorch::Pose2 &targetFrame) {
	_framesTracker->update<FRAME_TARGET> (targetFrame);
	emit (_targetUpdatedChannel);
}

void FramesTrackerModule::measuresTriggerSlot (const torch::Tensor &) {
	_framesTracker->updateMeasures ();
	emit (_measuresUpdatedChannel);
}

const char *resultStrings[] = {
	"none",
	"fail",
	"accept",
	"complete",
	"finalize"
};

void NapvigXModule::clockSlot ()
{
	if (!_napvigX.ready ())
		return;

	_napvigX.reset ();

	Policy::Type policy;
	Policy::Result result;

	result.type = Policy::RESULT_NONE;

	cout << "\e[33mNew Sample\e[0m" << endl;
	
	while (result.type != Policy::RESULT_ACCEPT) {
		policy = _napvigX.getNext (result.type);

		cout << "Current policy \e[32m" << _policies[policy]->name () << "\e[0m" << endl;

		result = callService<Policy::Result> (_policyChannels[policy], _napvigX.getInitialization ());
		cout << "Result \e[34m" << resultStrings[result.type] << "\e[0m" << endl;
	}

	// cout << "Accepted" << endl;

	_napvigX.finalize (result.command);
}

void NapvigXModule::abortSlot () {
	callService<Policy::Result> (_policyChannels[Policy::HALT], _napvigX.getInitialization ());
}

void LandmarksModule::poseUpdated () {
	_landmarksManager->update (_framesTracker->get (FRAME_MEASURES));
}

Policy::Result PolicyModule::followPolicySlot(const State &initialState)
{
	Policy::Result result = _policy->followPolicy (initialState);

	Tensor debugHistory = _policy->debugHistory ();
	Tensor debugCost = _policy->debugCost ();

	if (debugHistory.numel () > 0)
		emit (_historyChannel, debugHistory);

	if (debugCost.numel () > 0)
		emit (_costDebugChannel, debugCost);

	if (result.type == Policy::RESULT_ACCEPT)
		emit (_commandChannel, result.command);

	return result;
}

void PolicyModule::targetUpdated () {
	_policy->targetUpdated ();
}

void PolicyModule::measuresUpdated () {
	_policy->measuresUpdated ();
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

	Tensor commandInRobotFrame = _framesTracker->getIn (FRAME_ROBOT, FRAME_MEASURES, command);

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

void ProcessOutputs::debugCostSlot(const at::Tensor &cost) {
	emit (_tensorSink, cost, OUTPUT_VALUES);
}

void ProcessOutputs::targetSlot () {
	emit (_tensorSink, _framesTracker->get (FRAME_TARGET).translation ().coeffs, OUTPUT_TARGET);
}

void ProcessOutputs::measuresSlot (const torch::Tensor &measures)
{
	if (_params.toWorldFrame) {
		// not implemented
	}

	emit (_tensorSink, measures, OUTPUT_MEASURES);
}

void ProcessOutputs::poseSlot (const lietorch::Pose2 &frame) {
	emit (_poseSink, frame, OUTPUT_POSE_DEBUG);
}























