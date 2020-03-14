#include "CUDADFT.cuh"

__global__ void cudadft_kernel_dft(SignalBuffer_t device_buffer, SignalBuffer_t tmp)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	size_t size = get_max_buffer_size(tmp);
	cuComplex temp = make_cuComplex(0, 0);
	cuComplex sample, s;

	for (int i = 0; i < size; i++)
	{
		sample = get_sample(device_buffer, i);

		s = cuComplex_exp(-2.0f * PI * k * i / size);

		temp = cuCaddf(temp, cuCmulf(sample, s));
	}

	set_sample(tmp, k, temp);
}

__global__ void cudadft_kernel_copy(SignalBuffer_t device_buffer, SignalBuffer_t tmp)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	cuComplex sample = get_sample(tmp, k);
	set_sample(device_buffer, k, sample);
}


CUDADFT::CUDADFT(AbstractSignalProcessor* previous, size_t points) : CUDASignalProcessor(previous)
{
	this->points = points;
}

CUDADFT::~CUDADFT()
{
	cuda_deallocate_signal_buffer(this->tmp);
	CUDASignalProcessor::~CUDASignalProcessor();
}

void CUDADFT::init(size_t max_buffer_size)
{
	//cudaError_t status;
	//status = cuda_allocate_signal_buffer(this->tmp, max_buffer_size);
	//check_cuda_status(status);
	CUDASignalProcessor::init(max_buffer_size);
}

void CUDADFT::exec_kernel(SignalBuffer_t &host_buffer, SignalBuffer_t &device_buffer)
{
	cudaError_t status;

	if (get_buffer_size(host_buffer) <= 0)
		return;

	status = cuda_recreate_signal_buffer(device_buffer, points, 1);
	check_cuda_status(status);
	status = cuda_recreate_signal_buffer(tmp, points, 1);
	check_cuda_status(status);
	size_t readcount = get_max_buffer_size(device_buffer);

	dim3 threadsPerBlock;
	dim3 blocks;

	get_threads_blocks_count(readcount, threadsPerBlock, blocks);

	set_buffer_size(tmp, readcount);
	cudadft_kernel_dft << <blocks, threadsPerBlock >> > (device_buffer, tmp);
	wait_for_gpu();
	set_buffer_size(device_buffer, readcount);
	cudadft_kernel_copy << <blocks, threadsPerBlock >> > (device_buffer, tmp);

}