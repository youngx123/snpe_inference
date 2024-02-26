#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/IUserBufferFactory.hpp"
#include "DlSystem/TensorShape.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "./infer/nn_inference.h"

#include <iostream>
typedef unsigned int GLuint;

enum BUFFER_TYPE
{
	UNKNOWN,
	USERBUFFER_FLOAT,
	USERBUFFER_TF8,
	ITENSOR,
	//   ITENSOR,
	//   USERBUFFER_TF16
};

enum BUFFER_SRC_TYPE
{
	CPUBUFFER,
	GLBUFFER
};

class SnpeInferer : public NNInference
{
public:
	SnpeInferer();
	~SnpeInferer();

	bool Init(const std::string &model_path, size_t batch_size = 1,
			  InferDevice infer_device = InferDevice::kCPU,
			  std::vector<std::string> output_names = {},
			  bool profiling = false, const std::string &name = "Snpe") override;

	bool Init(const std::string &model_path, size_t batch_size = 1,
			  InferDevice infer_device = InferDevice::kCPU,
			  std::vector<std::string> output_names = {},
			  bool profiling = false, bool keep_ratio = false, int mean_std_offset = 0,
			  const std::string &name = "Snpe") override;

	bool Init(const std::string &model_path, zdl::DlSystem::Runtime_t runtime,
			  int mean_std_offset, std::vector<std::string> output_names);

	bool PreProcess(cv::Mat &img, int width, int height);
	// void DoInference(const std::vector<Input> &inputs, size_t batch_size = 1, bool preprocess = true) override;

	void DoInference(cv::Mat &inputs, size_t batch_size = 1, bool preprocess = true) override;

	std::vector<std::string> output_nbbinding_name()
	{
		return Statoutput_nbbinding_nameWarning_;
	}

	std::vector<size_t> getInputShape(const std::string &name);
	std::vector<size_t> getInputShape(const int index);
	std::vector<size_t> getOutputShape(const int index);
	std::vector<size_t> getOutputShape(const std::string &name);

	bool isInit() { return isInit_; }

	void InitOutputDimension();

	void CopyOutputToHost() override;
	void Synchronize() override;
	size_t input_count() const override;
	size_t input_width(size_t input_index) const override;
	size_t input_height(size_t input_index) const override;
	size_t output_size(size_t output_index) const override;
	size_t input_dimensions(size_t input_index = 0) const override;
	size_t output_dimensions(size_t output_index = 0) const override;
	size_t input_dimension(size_t input_index, size_t dim_index) const override;
	size_t output_dimension(size_t output_index, size_t dim_index) const override;

	std::string GetModelName() { return model_name_; }

public:
	// bool InitDialog();
	bool setOutputLayers(std::vector<std::string> &outputLayers);
	zdl::DlSystem::Runtime_t checkRuntime(zdl::DlSystem::Runtime_t runtime);

public:
	int input_height_;
	int input_width_;

	bool isInit_ = false;
	unsigned int batch_size_ = 1;
	int mean_std_offset_ = 0;
	bool staticQuantization_ = false;

	std::unique_ptr<zdl::DlContainer::IDlContainer> container_;
	std::unique_ptr<zdl::SNPE::SNPE> snpe_;
	zdl::DlSystem::Runtime_t runtime_;
	InferDevice infer_device_;
	zdl::DlSystem::StringList outputLayers_;

	std::vector<std::string> input_nbbinding_name_;
	std::vector<std::string> Statoutput_nbbinding_nameWarning_;

	std::map<std::string, std::vector<size_t>> m_inputShapes;
	std::map<std::string, std::vector<size_t>> m_outputShapes;

	std::vector<std::vector<size_t>> output_dimension_;

	std::vector<int> output_size_;

	std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> inputUserBuffers_;
	std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> outputUserBuffers_;
	zdl::DlSystem::UserBufferMap inputUserBufferMap_;
	zdl::DlSystem::UserBufferMap outputUserBufferMap_;
	std::unordered_map<std::string, std::vector<uint8_t>> applicationInputBuffers_;
	std::unordered_map<std::string, std::vector<uint8_t>> applicationOutputBuffers_;

	std::string model_name_;
};
