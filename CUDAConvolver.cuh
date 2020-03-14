#pragma once

#include "CUDASignalProcessor.cuh"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "CmplxUtils.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include "sndfile.h"

#include "LogUtils.h"

class CUDAConvolver : public CUDASignalProcessor
{
private:
	SignalBuffer_t tmp;
	SignalBuffer_t signal, device_signal;
	size_t temp_index;
	size_t samples_remaining;
public:
	CUDAConvolver(AbstractSignalProcessor* previous, SignalBuffer_t signal);
	~CUDAConvolver();
	void init(size_t max_buffer_size);
protected:
	void exec_kernel(SignalBuffer_t &host_buffer, SignalBuffer_t &device_buffer);
};



CUDAConvolver* create_cuda_convolver_from_file(AbstractSignalProcessor* previous, std::string filename, size_t conv_size);