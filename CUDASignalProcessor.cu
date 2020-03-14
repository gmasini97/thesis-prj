#include "CUDASignalProcessor.cuh"

#define MAX_THREADS_PER_BLOCK 1024

__global__ void cudasignalprocessor_copy(SignalBuffer_t dest, SignalBuffer_t src)
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	size_t src_size = get_buffer_size(src);
	cuComplex c;
	if (k < src_size)
		c = get_sample(src, k);
	else
		c = make_cuComplex(0,0);
	set_sample(dest, k, c);
}

void get_threads_blocks_count(size_t processes, dim3 &threadsPerBlock, dim3 &blocks)
{
	size_t tpb = processes > MAX_THREADS_PER_BLOCK ? MAX_THREADS_PER_BLOCK : processes;
	size_t r = processes % tpb > 0 ? 1 : 0;
	size_t b = processes / tpb + r;
	b = b == 0 ? 1 : b;

	threadsPerBlock = dim3(tpb);
	blocks = dim3(b);
}

cudaError_t cuda_recreate_signal_buffer(SignalBuffer_t& buffer, size_t datalen, int copy_data)
{
	cudaError_t status;
	size_t max_size = get_max_buffer_size(buffer);
	if (datalen > max_size)
	{
		size_t old_size = get_buffer_size(buffer);
		SignalBuffer_t newb;
		status = cuda_allocate_signal_buffer(newb, datalen);

		if (status != cudaSuccess)
			return status;

		if (copy_data){
			set_buffer_size(newb, datalen);

			dim3 tpb, blk;
			get_threads_blocks_count(datalen, tpb, blk);

			cudasignalprocessor_copy << <blk, tpb >> > (newb, buffer);

			status = cudaGetLastError();
			if (status != cudaSuccess)
				return status;

			status = cudaDeviceSynchronize();
			if (status != cudaSuccess)
				return status;
		}
		cuda_deallocate_signal_buffer(buffer);
		set_buffer_size(newb, old_size);
		buffer = newb;
	}
	return cudaSuccess;
}


cudaError_t transfer_buffer_host_to_device(SignalBuffer_t &device, SignalBuffer_t host) {
	cudaError_t status;
	size_t size = get_max_buffer_size(host);
	status = cuda_recreate_signal_buffer(device, size, 0);
	if (status != cudaSuccess)
		return status;
	status = cudaMemcpy(device.samples, host.samples, size * sizeof(device.samples[0]), cudaMemcpyHostToDevice);
	set_buffer_size(device, get_buffer_size(host));
	return status;
}

cudaError_t transfer_buffer_device_to_host(SignalBuffer_t &host, SignalBuffer_t device) {
	cudaError_t status;
	size_t size = get_max_buffer_size(device);
	recreate_signal_buffer(host, size, 0);
	status = cudaMemcpy(host.samples, device.samples, size * sizeof(host.samples[0]), cudaMemcpyDeviceToHost);
	set_buffer_size(host, get_buffer_size(device));
	return status;
}

cudaError_t cuda_allocate_signal_buffer(SignalBuffer_t &buffer, size_t datalen)
{
	cudaError_t status;

	cuComplex* samples;
	status = cudaMalloc((void**) & (samples), datalen * sizeof(cuComplex));
	if (status != cudaSuccess) return status;

	buffer.samples = samples;
	buffer.max_size = datalen;
	buffer.size = 0;

	return status;
}

cudaError_t cuda_clear_signal_buffer_deep(SignalBuffer_t &buffer)
{
	size_t max_size = get_max_buffer_size(buffer);

	cudaError_t status;
	
	cuComplex* tmp = new cuComplex[max_size];
	for (size_t i = 0 ; i < max_size; i++)
		tmp[i] = make_cuComplex(0,0);

	status = cudaMemcpy(buffer.samples, tmp, max_size * sizeof(cuComplex), cudaMemcpyHostToDevice);
	delete[] tmp;
	if (status != cudaSuccess)
		return status;

	set_buffer_size(buffer,0);

	return cudaSuccess;
}

void cuda_deallocate_signal_buffer(SignalBuffer_t &buffer)
{
	cudaFree(buffer.samples);
	buffer.size = 0;
	buffer.max_size = 0;
}

CUDASignalProcessor::CUDASignalProcessor(AbstractSignalProcessor* previous) : SignalProcessor(previous)
{
	this->err = 0;
	this->device_buffer = empty_signal_buffer();
}

CUDASignalProcessor::~CUDASignalProcessor()
{
	cuda_deallocate_signal_buffer(this->device_buffer);
	SignalProcessor::~SignalProcessor();
}

void CUDASignalProcessor::init(size_t max_buffer_size)
{
	cudaError_t status;

	status = cudaSetDevice(0);
	check_cuda_status(status, "set_device");

	//status = cuda_allocate_signal_buffer(this->device_buffer, max_buffer_size);
	//check_cuda_status(status, "alloc_dev_buffer");

	SignalProcessor::init(max_buffer_size);
}

int CUDASignalProcessor::check_cuda_status(cudaError_t status, const char* msg)
{
	if (status != cudaSuccess)
	{
		this->err = 1;
		std::cout << "[" << msg << "] cuda err: " << cudaGetErrorString(status) << std::endl;
		throw std::runtime_error(msg);
	}
	return this->err;
}

void CUDASignalProcessor::process_buffer(SignalBuffer_t &buffer)
{
	if (has_previous_processor())
		get_previous_processor()->process_buffer(buffer);

	LOG("CUDASignalProcessor process_buffer start\n");

	cudaError_t status;

	status = transfer_buffer_host_to_device(this->device_buffer, buffer);
	check_cuda_status(status, "buffer h>d");

	wait_for_gpu();

	this->exec_kernel(buffer, this->device_buffer);

	wait_for_gpu();

	status = transfer_buffer_device_to_host(buffer, this->device_buffer);
	check_cuda_status(status, "buffer d>h");

}

int CUDASignalProcessor::get_err()
{
	return this->err;
}

void CUDASignalProcessor::wait_for_gpu()
{
	cudaError_t status;
	status = cudaGetLastError();
	check_cuda_status(status, "last_err");

	status = cudaDeviceSynchronize();
	check_cuda_status(status, "dev_synch");
}