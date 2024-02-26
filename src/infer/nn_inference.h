#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
// #include "perception_pch.h"

// namespace calmcar
// {
// 	namespace oms
// 	{
// 		namespace perception
// 		{

enum class InferDevice : int
{
	kIGPU = 0, // nvidia intigrated GPU
	kDLA = 1,  // nvidia DLA
	kNPU = 2,  // hisi NPU, dsp
	kBPU = 3,  // Horizon BPU
	kCPU = -1  //
};

enum class DataLayout
{
	NHWC = 0, // NHWC
	NCHW = 1  // NCHW
};

struct Input
{
	enum ResizeType
	{
		Normal,
		WithRatio,
		Memcpy // Unused: test only
	};

	Input() = default;
	Input(void *ptr, size_t width, size_t height, size_t channel,
		  const cv::Rect &roi, ResizeType resize_type = Normal);
	~Input() = default;

	void *ptr = nullptr;
	size_t width = 0;
	size_t height = 0;
	size_t channel = 0;
	cv::Rect roi{};
	ResizeType resize_type = Normal;
};

class NNInference
{
public:
	NNInference() = default;
	virtual ~NNInference() = default;
	virtual bool Init(const std::string &model_path, size_t batch_size = 1,
					  InferDevice infer_device = InferDevice::kCPU,
					  std::vector<std::string> output_names = {},
					  bool profiling = false, const std::string &name = {});

	virtual bool Init(const std::string &model_path, size_t batch_size = 1,
					  InferDevice infer_device = InferDevice::kCPU,
					  std::vector<std::string> output_names = {},
					  bool profiling = false, bool keep_ratio = false,
					  int mean_std_offset = 0, const std::string &name = {});

	size_t batch_size() const;
	InferDevice device_type() const;
	bool profiling_enabled() const;

	// virtual void DoInference(const std::vector<Input> &inputs, size_t batch_size,
	// 						 bool preprocess = true) = 0;
	virtual void DoInference(cv::Mat &inputs, size_t batch_size = 1, bool preprocess = true) = 0;

	// Copy output from outputs_device_ to outputs_host_, does nothing by default
	virtual void CopyOutputToHost();

	// Synchronize device with host, does nothing by default
	virtual void Synchronize();

	// 1 by default
	virtual size_t input_count() const;
	// pixel count, 0 by default
	virtual size_t input_width(size_t input_index) const;
	// pixel count, 0 by default
	virtual size_t input_height(size_t input_index) const;
	/** default: batch_size * output_dimension[d0] * output_dimension[d1] * ...,
	 *  MUST be available and keep unchanged initialization. */
	virtual size_t output_size(size_t output_index) const;
	// pixel count, 0 by default
	virtual size_t output_width(size_t output_index) const;
	// pixel count, 0 by default
	virtual size_t output_height(size_t output_index) const;
	// 0 by default
	virtual size_t input_dimensions(size_t input_index) const;
	/** 0 by default, MUST be available and keep unchanged initialization. */
	virtual size_t output_dimensions(size_t output_index) const;
	// 0 by default
	virtual size_t input_dimension(size_t input_index, size_t dim_index) const;
	/** 0 by default, MUST be available and keep unchanged initialization. */
	virtual size_t output_dimension(size_t output_index, size_t dim_index) const;

	/** Output values inside CPU memory, default is outputs_host_, MUST be
	 *  available initialization, all pointer addresses MUST keep unchanged. */
	virtual std::vector<void *> &outputs_host();
	/** Output values inside Device memory, default is outputs_device_, MUST be
	 *  available initialization, all pointer addresses MUST keep unchanged. */
	virtual std::vector<void *> &outputs_device();
	// \overload outputs_host
	virtual const std::vector<void *> &outputs_host() const;
	// \overload outputs_host
	virtual const std::vector<void *> &outputs_device() const;

	// Model name or inferer name
	virtual const std::string &inferer_name() const;

	// binding name
	virtual std::vector<std::string> output_nbbinding_name();

	// Output data layout, default is nhwc
	virtual DataLayout output_data_layout() const;

	// Output data is quantilized, default is false
	virtual bool output_quantilized() const;

	// Copy partial data, not all platform supported
	virtual void CopyPartialDataToHost(int index = 0);

public:
	// Initialized by Init()
	size_t batch_size_ = 0;
	// Initialized by Init()
	InferDevice infer_device_ = InferDevice::kCPU;
	/** Initialized by Init(). It's just a flag, can be used for runtime analyzing
	 *  such as profile or tracing */
	bool profiling_ = false;
	// Should be allocated in initialization
	std::vector<void *> outputs_host_{};
	// Should be allocated in initialization
	std::vector<void *> outputs_device_{};
	// Model name or inferer name
	std::string inferer_name_{};

	bool keep_ratio_ = false;
	int mean_std_offset_ = 0;

private:
	NNInference(const NNInference &) = delete;
	NNInference &operator=(const NNInference &) = delete;
};

typedef std::shared_ptr<NNInference> InferenceHandle;

// 		} // namespace perception
// 	}	  // namespace oms
// } // namespace calmcar
