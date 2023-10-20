
#ifndef SNPE_INFER_H
#define SNPE_INFER_H

#include <SNPE/SNPE.hpp>
#include <DlContainer/IDlContainer.hpp>
#include <DlSystem/String.hpp>
#include <DlSystem/DlError.hpp>
#include <DlSystem/ITensor.hpp>
#include <DlSystem/ITensorFactory.hpp>
#include <SNPE/SNPEFactory.hpp>
#include <SNPE/SNPEBuilder.hpp>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <unordered_map>
#include <map>

typedef unsigned int GLuint;

enum InfereDevice
{
	CPU,
	GPU,
	DSP,
	APU
};

class SnpeInfer
{
public:
	SnpeInfer() = default;
	SnpeInfer(const std::string dlc_file, int dst_w, int dst_h, std::vector<std::string> output_node, int device = 0);
	// SnpeInfer(const std::string dlc_file, int dst_w, int dst_h, std::vector<std::string>output_node, int device=0);
	~SnpeInfer();
	zdl::DlSystem::Runtime_t checkRuntime(zdl::DlSystem::Runtime_t runtime);
	bool initDevice(int device);

	bool Init(const std::string &model_path, zdl::DlSystem::Runtime_t runtime,
			  std::vector<std::string> output_names);
	void showTime();
	bool PreProcess(const cv::Mat &img, int width, int height);
	void DoInference(const cv::Mat &inputs, size_t batch_size, bool preprocess);
	void InitOutputDimension();
	std::vector<size_t> getOutputShape(const std::string &name);
	size_t output_size(size_t output_index) const;

private:
	int count_num = 0;
	float total_time = 0.0f;
	float process_time = 0.0f;

private:
	int device_;
	std::string dlc_path;
	int input_w;
	int input_h;

private:
	std::vector<std::string> output_names;
	std::unique_ptr<zdl::DlContainer::IDlContainer> container_;
	std::unique_ptr<zdl::SNPE::SNPE> snpe_;

	zdl::DlSystem::Runtime_t runtime;

	std::vector<std::string> input_nbbinding_name_;
	std::vector<std::string> m_output_nbbinding_name;

	std::map<std::string, std::vector<size_t>> m_inputShapes;
	std::map<std::string, std::vector<size_t>> m_outputShapes;

	std::vector<float *> outputs_host_{};
	// std::vector<uint8_t *> outputs_host_int_{};
	std::vector<std::vector<size_t>> output_dimension_;
	std::vector<int> output_size_;

	std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> m_inputUserBuffers;
	std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> m_outputUserBuffers;
	zdl::DlSystem::UserBufferMap m_inputUserBufferMap;
	zdl::DlSystem::UserBufferMap m_outputUserBufferMap;
	std::unordered_map<std::string, std::vector<uint8_t>> m_applicationInputBuffers;
	std::unordered_map<std::string, std::vector<uint8_t>> m_applicationOutputBuffers;
	//

	// std::unique_ptr<zdl::SNPE::SNPE> snpe;
	// zdl::DlSystem::StringList outputLayers;
	// std::shared_ptr<zdl::DlSystem::ITensor> inTensor;
	// zdl::DlSystem::TensorMap outMap;
	// zdl::DlSystem::TensorMap inMap;
};

#endif // AI_DEMO_CLASSIFICATION_H
