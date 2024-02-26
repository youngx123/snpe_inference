#include "nn_inference.h"

// namespace calmcar
// {
// 	namespace oms
// 	{
// 		namespace perception
// 		{
float mean_[12] = {0.0f, 0.0f, 0.0f, 123.461f, 112.923f, 118.8465f,
				   127.5f, 127.5f, 127.5f, 0.0f, 0.0f, 0.0f};
float scale_[12] = {
	1.0f,
	1.0f,
	1.0f,
	1.0f / 61.8468f,
	1.0f / 51.8647f,
	1.0f / 57.36741f,
	1.0f / 127.5f,
	1.0f / 127.5f,
	1.0f / 127.5f,
	1.0f / 255.0f,
	1.0f / 255.0f,
	1.0f / 255.0f,
};

Input::Input(void *ptr, size_t width, size_t height, size_t channel,
			 const cv::Rect &roi, ResizeType resize_type)
	: ptr(ptr),
	  width(width),
	  height(height),
	  channel(channel),
	  roi(roi),
	  resize_type(resize_type) {}

bool NNInference::Init(const std::string &model_path, size_t batch_size,
					   InferDevice infer_device,
					   std::vector<std::string> output_names, bool profiling,
					   const std::string &name)
{
	batch_size_ = batch_size;
	infer_device_ = infer_device;
	profiling_ = profiling;
	inferer_name_ = name;
	return true;
}

bool NNInference::Init(const std::string &model_path, size_t batch_size,
					   InferDevice infer_device,
					   std::vector<std::string> output_names, bool profiling,
					   bool keep_ratio, int mean_std_offset,
					   const std::string &name)
{
	batch_size_ = batch_size;
	infer_device_ = infer_device;
	profiling_ = profiling;
	keep_ratio_ = keep_ratio;
	mean_std_offset_ = mean_std_offset;
	inferer_name_ = name;
	return true;
}

size_t NNInference::batch_size() const { return batch_size_; }

InferDevice NNInference::device_type() const { return infer_device_; }

bool NNInference::profiling_enabled() const { return profiling_; }

void NNInference::CopyOutputToHost() {}

void NNInference::Synchronize() {}

size_t NNInference::input_count() const { return 1; }

size_t NNInference::input_width(size_t) const { return 0; }

size_t NNInference::input_height(size_t) const { return 0; }

size_t NNInference::output_size(size_t output_index) const
{
	if (output_index >= outputs_device_.size())
	{
		return 0;
	}
	size_t size = batch_size_;
	for (size_t i = 0; i < output_dimensions(output_index); ++i)
	{
		size *= output_dimension(output_index, i);
	}

	return size;
}

size_t NNInference::output_width(size_t) const { return 0; }

size_t NNInference::output_height(size_t) const { return 0; }

size_t NNInference::input_dimensions(size_t) const { return 0; }

size_t NNInference::output_dimensions(size_t) const { return 0; }

size_t NNInference::input_dimension(size_t, size_t) const { return 0; }

size_t NNInference::output_dimension(size_t, size_t) const { return 0; }

std::vector<void *> &NNInference::outputs_host() { return outputs_host_; }

std::vector<void *> &NNInference::outputs_device() { return outputs_device_; }

const std::vector<void *> &NNInference::outputs_host() const
{
	return outputs_host_;
}

const std::vector<void *> &NNInference::outputs_device() const
{
	return outputs_device_;
}

const std::string &NNInference::inferer_name() const { return inferer_name_; }

std::vector<std::string> NNInference::output_nbbinding_name() { return {}; }

DataLayout NNInference::output_data_layout() const { return DataLayout::NHWC; }

bool NNInference::output_quantilized() const { return false; }

void NNInference::CopyPartialDataToHost(int index)
{
	CopyOutputToHost();
	return;
}

// 		} // namespace perception
// 	}	  // namespace oms
// } // namespace calmcar
