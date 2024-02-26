#include <opencv2/opencv.hpp>
#include <typeinfo>
#include "snpe_infer.h"

#include "DiagLog/IDiagLog.hpp"
#include "utils/createBuffer.h"

// extern float mean_[];
// extern float scale_[];

bool SnpeInferer::Init(const std::string &model_path, size_t batch_size,
					   InferDevice infer_device, std::vector<std::string> output_names,
					   bool profiling, const std::string &name)
{
	model_name_ = model_path;
	infer_device_ = infer_device;
	zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
	if (InferDevice::kIGPU == infer_device_)
	{
		runtime = zdl::DlSystem::Runtime_t::GPU;
		printf("****####****runtime  on GPU device ****####****\n");
	}
	else if (InferDevice::kNPU == infer_device_)
	{
		runtime = zdl::DlSystem::Runtime_t::DSP;
		printf("*************** runtime  on DSP device  ***************\n");
	}
	else
	{
		printf("############  runtime  on cpu device #############\n");
	}
	return SnpeInferer::Init(model_path, runtime, 0, output_names);
}

bool SnpeInferer::Init(const std::string &model_path, size_t batch_size,
				InferDevice infer_device, std::vector<std::string> output_names,
				bool keep_ratio, bool profiling, int mean_std_offset, const std::string &name)
{
	model_name_ = model_path;
	infer_device_ = infer_device;
	zdl::DlSystem::Runtime_t runtime; // = zdl::DlSystem::Runtime_t::CPU;
	if (InferDevice::kIGPU == infer_device_)
	{
		runtime = zdl::DlSystem::Runtime_t::GPU;
		printf("****####****runtime  on GPU device ****####****\n");
	}
	else if (InferDevice::kNPU == infer_device_)
	{
		runtime = zdl::DlSystem::Runtime_t::DSP;
		printf("****####****runtime  on DSP device ****####****\n");
	}
	else if (InferDevice::kCPU == infer_device_)
	{
		runtime = zdl::DlSystem::Runtime_t::CPU;
		printf("****####****runtime  on CPU device ****####****\n");
	}
	return SnpeInferer::Init(model_path, runtime, mean_std_offset, output_names);
}

bool SnpeInferer::Init(const std::string &model_path, zdl::DlSystem::Runtime_t runtime,
					   int mean_std_offset, std::vector<std::string> output_names)
{
	model_name_ = model_path;
	mean_std_offset_ = 3 * mean_std_offset;
	runtime_ = runtime;
	checkRuntime(runtime);
	printf("checking runtime\n");
	zdl::DlSystem::PerformanceProfile_t profile = zdl::DlSystem::PerformanceProfile_t::BURST;

	container_ = zdl::DlContainer::IDlContainer::open(model_path);
	if (container_ == nullptr)
	{
		printf("Error while opening the container file : %s \n", model_path.c_str());
		std::exit(EXIT_FAILURE);
	}

	zdl::SNPE::SNPEBuilder snpeBuilder(container_.get());
	zdl::DlSystem::PlatformConfig platformConfig;
	// platformConfig.setPlatformOptions("unsignedPD:ON");

	zdl::DlSystem::StringList outputTensorNames;
	for (auto str : output_names)
	{
		outputTensorNames.append(str.c_str());
	}
	printf("checking runtimeList\n");
	zdl::DlSystem::RuntimeList runtimeList;
	runtimeList.add(runtime_);
	if (runtime_ != zdl::DlSystem::Runtime_t::CPU)
	{
		runtimeList.add(zdl::DlSystem::Runtime_t::CPU);
	}

	snpe_ = snpeBuilder.setOutputLayers({})
				.setOutputTensors(outputTensorNames)
				.setRuntimeProcessorOrder(runtimeList)
				.setPerformanceProfile(profile)
				.setPlatformConfig(platformConfig)
				.setProfilingLevel(zdl::DlSystem::ProfilingLevel_t::OFF)
				//  .setCPUFallbackMode(false)
				.setUseUserSuppliedBuffers(true)
				.setInitCacheMode(true)
				.build();
	printf("snpe _ \n");
	if (nullptr == snpe_.get())
	{
		const char *errStr = zdl::DlSystem::getLastErrorString();
		printf("SNPE build failed: {%s} \n", errStr);
		std::exit(1);
		return false;
	}
	// get input tensor names of the network that need to be populated
	const auto &inputNamesOpt = snpe_->getInputTensorNames();
	if (!inputNamesOpt)
	{
		throw std::runtime_error("Error obtaining input tensor names");
	}

	const zdl::DlSystem::StringList &inputNames = *inputNamesOpt;
	// create SNPE user buffers for each application storage buffer
	for (const char *name : inputNames)
	{
		printf("snpe _ %s\n", name);
		input_nbbinding_name_.push_back(name);
		// get attributes of buffer by name
		auto bufferAttributesOpt = snpe_->getInputOutputBufferAttributes(name);
		if (!bufferAttributesOpt)
		{
			printf("Error obtaining attributes for input tensor: %s \n", name);
			return false;
		}

		const zdl::DlSystem::TensorShape &bufferShape = (*bufferAttributesOpt)->getDims();
		std::vector<size_t> tensorShape;
		for (size_t j = 0; j < bufferShape.rank(); j++)
		{
			printf("tensorShape %d \n", bufferShape[j]);
			tensorShape.push_back(bufferShape[j]);
		}
		m_inputShapes.emplace(name, tensorShape);
	}
	// get output tensor names of the network that need to be populated
	const auto &outputNamesOpt = snpe_->getOutputTensorNames();
	// const auto& outputNamesOpt = snpe_->getOutputLayerNames();

	if (!outputNamesOpt)
		throw std::runtime_error("Error obtaining output tensor names");
	const zdl::DlSystem::StringList &outputNames = *outputNamesOpt;

	// create SNPE user buffers for each application storage buffer
	for (const char *name : outputNames)
	{
		printf("outputNames _ %s\n", name);
		Statoutput_nbbinding_nameWarning_.push_back(name);
		// get attributes of buffer by name
		auto bufferAttributesOpt = snpe_->getInputOutputBufferAttributes(name);
		if (!bufferAttributesOpt)
		{
			printf("Error obtaining attributes for input tensor: %s\n", name);
			return false;
		}

		const zdl::DlSystem::TensorShape &bufferShape = (*bufferAttributesOpt)->getDims();
		std::vector<size_t> tensorShape;
		for (size_t j = 0; j < bufferShape.rank(); j++)
		{
			printf("tensorShape %d \n", bufferShape[j]);
			tensorShape.push_back(bufferShape[j]);
		}
		m_outputShapes.emplace(name, tensorShape);
	}
	printf("createOutputBufferMap end  \n");
	createOutputBufferMap(outputUserBufferMap_, applicationOutputBuffers_, outputUserBuffers_, snpe_, false, 0);
	createInputBufferMap(inputUserBufferMap_, applicationInputBuffers_, inputUserBuffers_, snpe_, false, 0);

	// createInputBufferMap(inputUserBufferMap_, applicationInputBuffers_, inputUserBuffers_, snpe_, true, 8);
	printf("createInputBufferMap end  \n");
	isInit_ = true;

	auto input_shape = getInputShape(0);
	if (input_shape.size())
	{
		batch_size_ = input_shape[0];
	}
	// printf("batch_size_ end  \n");
	// InitDialog();
	input_width_ = input_width(0);
	input_height_ = input_height(0);
	printf("input dimention :   %d, %d \n", input_width_, input_height_);
	InitOutputDimension();

	return true;
}

void SnpeInferer::InitOutputDimension()
{
	if (!Statoutput_nbbinding_nameWarning_.size())
	{
		printf("Statoutput_nbbinding_nameWarning_ is null!");
		return;
	}

	outputs_host_.resize(Statoutput_nbbinding_nameWarning_.size());
	int out_memory_size = 0;

	for (auto output_name : Statoutput_nbbinding_nameWarning_)
	{
		auto bufferPtr = outputUserBufferMap_.getUserBuffer(output_name.c_str());
		if (nullptr == bufferPtr)
		{
			printf("Faild to find output buffer name %s.",
				   output_name.c_str());
		}
		auto output_dims = getOutputShape(output_name);
		output_dimension_.push_back(output_dims);
		output_size_.push_back(bufferPtr->getSize() / sizeof(float));
		out_memory_size += bufferPtr->getSize();
	}

	outputs_host_[0] = new float[out_memory_size];
	for (int i = 1; i < outputs_host_.size(); ++i)
	{
		outputs_host_[i] = (float *)outputs_host_[i - 1] + output_size(i - 1);
	}
}

void SnpeInferer::CopyOutputToHost() {}

void SnpeInferer::Synchronize() {}

size_t SnpeInferer::input_count() const { return 0; }

size_t SnpeInferer::input_width(size_t input_index) const
{
	size_t width = 0;
	std::string input_name = "";
	if (input_nbbinding_name_.size() > input_index)
	{
		input_name = input_nbbinding_name_[input_index];
	}
	if (m_inputShapes.find(input_name) != m_inputShapes.end())
	{
		switch (m_inputShapes.at(input_name).size())
		{
		case 0:
		case 1:
		case 2:
			width = 0;
			break;
		case 3:
			width = m_inputShapes.at(input_name)[1];
			break;
		case 4:
			width = m_inputShapes.at(input_name)[2];
			break;
		default:
			width = 0;
			break;
		}
	}
	return width;
}

size_t SnpeInferer::input_height(size_t input_index) const
{
	size_t height = 0;
	std::string input_name = "";
	if (input_nbbinding_name_.size() > input_index)
	{
		input_name = input_nbbinding_name_[input_index];
	}
	if (m_inputShapes.find(input_name) != m_inputShapes.end())
	{
		switch (m_inputShapes.at(input_name).size())
		{
		case 0:
		case 1:
		case 2:
			height = 0;
			break;
		case 3:
			height = m_inputShapes.at(input_name)[0];
			break;
		case 4:
			height = m_inputShapes.at(input_name)[1];
			break;
		default:
			height = 0;
			break;
		}
	}
	return height;
}

size_t SnpeInferer::output_size(size_t output_index) const
{
	return output_size_[output_index];
}

size_t SnpeInferer::input_dimensions(size_t input_index) const
{
	size_t input_dims = 0;
	return input_dims;
}

size_t SnpeInferer::output_dimensions(size_t output_index) const
{
	if (output_index + 1 > output_dimension_.size())
	{
		return 0;
	}
	else
	{
		return output_dimension_[output_index].size();
	}
}

size_t SnpeInferer::input_dimension(size_t input_index,
									size_t dim_index) const
{
	// size_t input_dims = 0;
	// if (input_dimension_.size() < input_index + 1)
	// {
	// 	printf("input_index  is invalid! ");
	// 	return input_dims;
	// }
	// else if (input_dimension_[input_index].size() < dim_index + 1)
	// {
	// 	printf("dim_index  is invalid! ");
	// 	return input_dims;
	// }
	// input_dims = input_dimension_[input_index][dim_index];
	// return input_dims;
	return 0;
}

size_t SnpeInferer::output_dimension(size_t output_index,
									 size_t dim_index) const
{
	size_t output_dims = 0;
	if (output_dimension_.size() < output_index + 1)
	{
		printf("output_index  is invalid! ");
		return 0;
	}
	else if (output_dimension_[output_index].size() < dim_index + 1)
	{
		printf("dim_index  is invalid! ");
		return 0;
	}
	output_dims = output_dimension_[output_index][dim_index];
	return output_dims;
}

zdl::DlSystem::Runtime_t SnpeInferer::checkRuntime(
	zdl::DlSystem::Runtime_t runtime)
{
	static zdl::DlSystem::Version_t Version =
		zdl::SNPE::SNPEFactory::getLibraryVersion();

	// Print Version number
	printf("SNPE Version: %s\n", Version.asString().c_str());
	if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime))
	{
		printf("Selected runtime not present.\n");
		std::exit(EXIT_FAILURE);
	}

	return runtime;
}

SnpeInferer::SnpeInferer() {}

SnpeInferer::~SnpeInferer()
{
	if (nullptr != snpe_)
	{
		snpe_.reset();
		snpe_ = nullptr;
	}
	if (outputs_host_.size() && outputs_host_[0] != nullptr)
	{
		delete[] outputs_host_[0];
		for (int i = 0; i < outputs_host_.size(); i++)
		{
			outputs_host_[i] = nullptr;
		}
	}
}

// bool SnpeInferer::InitDialog()
// {
// 	// Configure logging output and start logging.The snpe -
// 	//     diagview executable can be used to read the content
// 	//         of this diagnostics file
// 	auto logger_opt = snpe_->getDiagLogInterface();
// 	if (!logger_opt)
// 		throw std::runtime_error("SNPE failed to obtain logging interface");
// 	auto logger = *logger_opt;
// 	auto opts = logger->getOptions();
// 	static std::string OutputDir = "./log_output/";
// 	opts.LogFileDirectory = OutputDir;
// 	if (!logger->setOptions(opts))
// 	{
// 		std::cerr << "Failed to set options" << std::endl;
// 		return false;
// 	}
// 	if (!logger->start())
// 	{
// 		std::cerr << "Failed to start logger" << std::endl;
// 		return false;
// 	}
// 	return true;
// }

bool SnpeInferer::setOutputLayers(std::vector<std::string> &outputLayers)
{
	for (size_t i = 0; i < outputLayers.size(); i++)
	{
		outputLayers_.append(outputLayers[i].c_str());
	}

	return true;
}

std::vector<size_t> SnpeInferer::getInputShape(const std::string &name)
{
	if (isInit())
	{
		if (m_inputShapes.find(name) != m_inputShapes.end())
		{
			return m_inputShapes.at(name);
		}
		printf("Can't find any input layer named %s \n", name.c_str());
		return {};
	}
	else
	{
		printf(
			"The getInputShape() needs to be called after AICContext is initialized! \n");
		return {};
	}
}

std::vector<size_t> SnpeInferer::getInputShape(const int index)
{
	std::string name;
	if (input_nbbinding_name_.size() <= index)
	{
		return {};
	}
	else
	{
		name = input_nbbinding_name_[index];
	}
	return getInputShape(name);
}

std::vector<size_t> SnpeInferer::getOutputShape(const std::string &name)
{
	if (isInit())
	{
		if (m_outputShapes.find(name) != m_outputShapes.end())
		{
			return m_outputShapes.at(name);
		}
		printf("Can't find any ouput layer named %s\n", name.c_str());
		return {};
	}
	else
	{
		printf(
			"The getOutputShape() needs to be called after AICContext is  initialized! \n");
		return {};
	}
}

std::vector<size_t> SnpeInferer::getOutputShape(const int index)
{
	std::string name;
	if (Statoutput_nbbinding_nameWarning_.size() <= index)
	{
		return {};
	}
	else
	{
		name = Statoutput_nbbinding_nameWarning_[index];
	}
	return getOutputShape(name);
}

bool SnpeInferer::PreProcess(cv::Mat &img, int width, int height)
{
	size_t batch = 1;
	size_t inputHeight = height;
	size_t inputWidth = width;
	size_t channel = 3;

	// CALMCAR_INFO("InputShape:%d,%d,%d,%d", (int)batch, (int)inputHeight,
	//              (int)inputWidth, (int)channel);

	if (img.empty())
	{
		printf("Invalid image! \n");
		return false;
	}
	cv::Mat image_trans;
	// cv::resize(img, image_trans, cv::Size(inputWidth, inputHeight));
	int size = img.rows * img.cols * img.channels();
	img.convertTo(image_trans, CV_32F);
	// std::vector<float> input_data;
	// for (int i = 0; i < size; i++) {
	//   float output = (*(image_trans.data + i));
	//   input_data.push_back(output);
	// }

	for (int i = 0; i < input_nbbinding_name_.size(); i++)
	{
		// memcpy(&applicationInputBuffers_.at(input_nbbinding_name_[i].c_str())[0],
		//        &input_data[0], size * sizeof(input_data[0]));
		memcpy(&applicationInputBuffers_.at(input_nbbinding_name_[i].c_str())[0],
			   image_trans.data, size * sizeof(float));
	}

	return true;
}

/*
void SnpeInferer::DoInference(const std::vector<Input> &inputs,
							  size_t batch_size, bool preprocess)
{
	if (inputs.empty() || !inputs[0].ptr)
	{
		printf("inputs is NULL!");
		return;
	}

	cv::Mat image =
		cv::Mat(inputs[0].height, inputs[0].width, CV_8UC3, inputs[0].ptr);

	auto roi = inputs[0].roi;

	if (roi.x < 0)
	{
		printf("roi.x:%d", roi.x);
		roi.x = 0;
	}
	if (roi.y < 0)
	{
		printf("roi.y:%d", roi.y);

		roi.y = 0;
	}
	if (roi.x + roi.width > image.cols)
	{
		printf("input image width:%d", image.cols);
		printf("input roi:(x)%d, (width)%d", roi.x, roi.width);
		roi.width = image.cols - roi.x;
		printf("model_name:%s", model_name_.c_str());
	}
	if (roi.y + roi.height > image.rows)
	{
		printf("input image height:%d", image.rows);
		printf("input roi:(y)%d, (height)%d", roi.y, roi.height);
		roi.height = image.rows - roi.y;
		printf("model_name:%s", model_name_.c_str());
	}

	if (roi.width <= 0 || roi.height <= 0)
	{
		printf("x,y,width,height:%d,%d,%d,%d", roi.x, roi.y, roi.width,
			   roi.height);
		printf("model_name:%s", model_name_.c_str());
		printf("Exit DoInference!");
		return;
	}

	cv::Mat image_roi = image(roi);
	if (preprocess)
	{
		PreProcess(image_roi, input_width_, input_height_);
	}

	if (!snpe_->execute(inputUserBufferMap_, outputUserBufferMap_))
	{
		printf("SnpeInferer execute failed(%s): %s", model_name_.c_str(),
			   zdl::DlSystem::getLastErrorString());
		return;
	}

	std::map<std::string, std::vector<float>> output_result;
	int index = 0;
	for (auto name : Statoutput_nbbinding_nameWarning_)
	{
		// CALMCAR_INFO("name:%s", name.c_str());
		auto bufferPtr = outputUserBufferMap_.getUserBuffer(name.c_str());
		if (nullptr == bufferPtr)
		{
			printf("Faild to find output buffer name %s.\n", name.c_str());
		}

		memcpy(outputs_host_[index], &applicationOutputBuffers_.at(name)[0],
			   bufferPtr->getSize());

		index++;
	}

	return;
}
*/

void SnpeInferer::DoInference(cv::Mat &inputs, size_t batch_size, bool preprocess)
{
	if (inputs.empty())
	{
		printf("inputs is NULL! \n");
		return;
	}

	cv::Mat image_roi = inputs.clone();
	if (preprocess)
	{
		PreProcess(image_roi, input_width_, input_height_);
	}

	if (!snpe_->execute(inputUserBufferMap_, outputUserBufferMap_))
	{
		printf("SnpeInferer execute failed(%s): %s \n", model_name_.c_str(), zdl::DlSystem::getLastErrorString());
		return;
	}

	std::map<std::string, std::vector<float>> output_result;
	int index = 0;
	for (auto name : Statoutput_nbbinding_nameWarning_)
	{
		// CALMCAR_INFO("name:%s", name.c_str());
		auto bufferPtr = outputUserBufferMap_.getUserBuffer(name.c_str());
		if (nullptr == bufferPtr)
		{
			printf("Faild to find output buffer name %s.\n", name.c_str());
		}

		memcpy(outputs_host_[index], &applicationOutputBuffers_.at(name)[0], bufferPtr->getSize());

		index++;
	}

	return;
}
