#pragma once
#include <memory>
#include "nn_inference.h"

class InferenceFactory
{
public:
	static InferenceHandle CreateInference(void);
};
