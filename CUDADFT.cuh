#pragma once

#include "CUDASignalProcessor.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "CmplxUtils.cuh"

#include "LogUtils.h"

class CUDADFT : public CUDASignalProcessor
{
private:
	size_t points;
	SignalBuffer_t tmp = empty_signal_buffer();
public:
	CUDADFT(AbstractSignalProcessor* previous, size_t points);
	~CUDADFT();
	void init(size_t max_buffer_size);
protected:
	void exec_kernel(SignalBuffer_t &host_buffer, SignalBuffer_t &device_buffer);
};