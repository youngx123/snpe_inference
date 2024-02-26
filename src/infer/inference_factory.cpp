#include "inference_factory.h"

#include <memory>

#include "snpe/snpe_infer.h"
using InferImpl_ = SnpeInferer;

InferenceHandle InferenceFactory::CreateInference(void)
{
	InferenceHandle infer;
	infer = std::shared_ptr<InferImpl_>(new InferImpl_());
	return infer;
}
