#pragma once

#include "SignalProcessor.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>
#include "LogUtils.h"


cudaError_t transfer_buffer_host_to_device(SignalBuffer_t &device, SignalBuffer_t host);
cudaError_t transfer_buffer_device_to_host(SignalBuffer_t &host, SignalBuffer_t device);

cudaError_t cuda_allocate_signal_buffer(SignalBuffer_t &buffer, size_t datalen);
cudaError_t cuda_recreate_signal_buffer(SignalBuffer_t& buffer, size_t datalen, int copy_data);

cudaError_t cuda_clear_signal_buffer_deep(SignalBuffer_t &buffer);

void get_threads_blocks_count(size_t processes, dim3 &threadsPerBlock, dim3 &blocks);
void cuda_deallocate_signal_buffer(SignalBuffer_t &buffer);


class CUDASignalProcessor : public SignalProcessor
{
private:
	SignalBuffer_t device_buffer;
	int err;
protected:
	int check_cuda_status(cudaError_t status, const char* msg = "");
	void wait_for_gpu();
	virtual void exec_kernel(SignalBuffer_t &host_buffer, SignalBuffer_t &device_buffer) = 0;
public:
	CUDASignalProcessor(AbstractSignalProcessor* previous);
	~CUDASignalProcessor();
	void init(size_t max_buffer_size);
	void process_buffer(SignalBuffer_t &buffer);
	int get_err();
};