#include "CUDAConvolver.cuh"

#define SIGNAL_CHANNEL 0

__device__ size_t cuda_bounded_index(SignalBuffer_t buffer, size_t index)
{
	size_t channel_size = get_buffer_size(buffer);
	return index >= channel_size ? index % channel_size : index;
}

size_t bounded_index(size_t max, size_t index)
{
	return index >= max ? index % max : index;
}

__global__ void cudaconvolver_kernel_output(SignalBuffer_t device_buffer, SignalBuffer_t signal, SignalBuffer_t tmp, size_t temp_index)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	size_t out_size = get_buffer_size(tmp);
	if (k >= out_size)
		return;

	size_t signal_size = get_buffer_size(signal);

	cuComplex temp = make_cuComplex(0, 0);
	cuComplex signal_sample, input_sample;

	size_t index = cuda_bounded_index(tmp, temp_index + k);
	cuComplex temp_sample = get_sample(tmp, index);

	for (int i = 0; i < signal_size; i++)
	{
		signal_sample = get_sample(signal, i);
		if (i > k)
			input_sample = make_cuComplex(0,0);
		else
			input_sample = get_sample(device_buffer, k-i);
		temp_sample = cuCaddf(temp_sample, cuCmulf(signal_sample, input_sample));
	}
	set_sample(tmp, index, temp_sample);
}

__global__ void cudaconvolver_kernel_copy(SignalBuffer_t device_buffer, SignalBuffer_t tmp, size_t temp_index)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	size_t tmp_size = get_buffer_size(tmp);
	size_t out_size = get_buffer_size(device_buffer);
	if (k >= tmp_size)
		return;

	size_t index = cuda_bounded_index(tmp, temp_index + k);
	cuComplex sample = get_sample(tmp, index);
	set_sample(device_buffer, k, sample);
	if (k < out_size)
		set_sample(tmp, index, make_cuComplex(0,0));
}









CUDAConvolver::CUDAConvolver(AbstractSignalProcessor* previous, SignalBuffer_t signal) : CUDASignalProcessor(previous)
{
	this->signal = signal;
	this->temp_index = 0;
	this->samples_remaining = 0;
	this->tmp = empty_signal_buffer();
	this->device_signal = empty_signal_buffer();
}

CUDAConvolver::~CUDAConvolver()
{
	delete_signal_buffer(this->signal);
	cuda_deallocate_signal_buffer(this->tmp);
	cuda_deallocate_signal_buffer(this->device_signal);
	CUDASignalProcessor::~CUDASignalProcessor();
}

void CUDAConvolver::init(size_t max_buffer_size)
{
	cudaError_t status;
	size_t extra_size = get_max_buffer_size(this->signal) - 1;
/*
	status = cuda_allocate_signal_buffer(this->tmp, max_buffer_size + extra_size);
	check_cuda_status(status);

	status = cuda_clear_signal_buffer_deep(this->tmp);
	check_cuda_status(status);
*/
	status = cuda_allocate_signal_buffer(this->device_signal, get_max_buffer_size(signal));
	check_cuda_status(status);

	transfer_buffer_host_to_device(this->device_signal, this->signal);
	CUDASignalProcessor::init(max_buffer_size);
}

void CUDAConvolver::exec_kernel(SignalBuffer_t &host_buffer, SignalBuffer_t &device_buffer)
{
	LOG("CUDAConvolver kernel start\n");

	cudaError_t status;

	cudaStream_t stream;
	size_t buffer_size = get_buffer_size(host_buffer);
	size_t signal_size = get_buffer_size(signal);
	size_t outcount = buffer_size + signal_size - 1;

	status = cuda_recreate_signal_buffer(tmp, outcount, 1);
	check_cuda_status(status, "temporary cuda buffer realloc");

	LOG("CUDAConvolver starting stream %lli\n", channel);

	status = cudaStreamCreate(&stream);
	check_cuda_status(status, "stream create");

	size_t bounded_max = get_max_buffer_size(tmp);
	set_buffer_size(tmp, bounded_max);

	if ((buffer_size == 0 || signal_size == 0))
	{
		if (samples_remaining > 0){
			size_t max_size = get_max_buffer_size(host_buffer);
			size_t to_read = max_size < samples_remaining ? max_size : samples_remaining;
			size_t tmp_size = get_max_buffer_size(tmp);
			tmp_size = tmp_size < to_read + temp_index ? tmp_size : to_read+temp_index;

			dim3 threadsPerBlock;
			dim3 blocks;
			get_threads_blocks_count(to_read, threadsPerBlock, blocks);

			set_buffer_size(device_buffer, to_read);
			cudaconvolver_kernel_copy << <blocks, threadsPerBlock, 0, stream >> > (device_buffer, this->tmp, temp_index);

			this->temp_index = bounded_index(bounded_max, temp_index + to_read);
			this->samples_remaining = samples_remaining - to_read;
		}
		return;
	}

	dim3 threadsPerBlock;
	dim3 blocks;
	get_threads_blocks_count(outcount, threadsPerBlock, blocks);

	cudaconvolver_kernel_output << <blocks, threadsPerBlock, 0, stream >> > (device_buffer, this->device_signal, this->tmp, temp_index);
	cudaconvolver_kernel_copy << <blocks, threadsPerBlock, 0, stream >> > (device_buffer, this->tmp, temp_index);

	LOG("CUDAConvolver started stream %lli\n", channel);

	this->temp_index = bounded_index(outcount, temp_index + buffer_size);
	this->samples_remaining = signal_size - 1;

	if (stream != NULL) {
		LOG("CUDAConvolver waiting stream\n");
		status = cudaStreamSynchronize(stream);
		check_cuda_status(status, "stream sync");
		status = cudaStreamDestroy(stream);
		check_cuda_status(status, "stream destr");
		LOG("CUDAConvolver destroyed stream\n");
	}
}








CUDAConvolver* create_cuda_convolver_from_file(AbstractSignalProcessor* previous, std::string filename, size_t conv_size)
{
	SF_INFO info;
	memset(&info, 0, sizeof(SF_INFO));
	SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &info);

	if (info.channels != 1) {
		std::cout << "only 1 channel convolution kernel allowed" << std::endl;
		return NULL;
	}

	float* real = new float[conv_size];

	size_t actual_read = sf_read_float(file, real, conv_size);

	SignalBuffer_t buffer = create_signal_buffer(conv_size);
	signal_buffer_from_real_floats(buffer, real, actual_read);

	sf_close(file);
	delete[] real;

	return new CUDAConvolver(previous, buffer);
}