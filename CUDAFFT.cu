#include "CUDAFFT.cuh"

__device__ void cuda_butterfly_calculation(cuComplex* a, cuComplex* b, cuComplex w)
{
	cuComplex aa = *a;
	cuComplex bw = cuCmulf(*b, w);

	*a = cuCaddf(aa, bw);
	*b = cuCsubf(aa, bw);
}

__device__ __host__ size_t reverse_bits(size_t x, size_t bits)
{
	size_t rev = 0;
	while (x && bits)
	{
		rev = (rev << 1) | (x & 1);
		x = x >> 1;
		bits --;
	}
	rev = rev << bits;
	return rev;
}

__global__ void cudafft_kernel_br_sort(SignalBuffer_t device_buffer, SignalBuffer_t tmp, size_t bits)
{
	unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int rev = reverse_bits(k, bits);

	if (k > rev)
		return;

	cuComplex cmplx1 = get_sample(device_buffer, k);
	cuComplex cmplx2 = get_sample(device_buffer, rev);
	set_nr_sample(tmp, rev, cmplx1);
	set_nr_sample(tmp, k, cmplx2);
}

__global__ void cudafft_kernel_butterflies(SignalBuffer_t in_buf, SignalBuffer_t out_buf, size_t level)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	size_t bpd = (size_t)powf(2, level);
	int index_a = k + (size_t)(k / bpd) * bpd;
	int index_b = index_a + bpd;
	cuComplex a = get_sample(in_buf, index_a);
	cuComplex b = get_sample(out_buf, index_b);
	cuComplex w = cuComplex_exp(-(2*PI*index_a)/(bpd*2));
	cuda_butterfly_calculation(&a, &b, w);
	set_nr_sample(out_buf, index_a, a);
	set_nr_sample(out_buf, index_b, b);
}

__global__ void cudafft_kernel_copy(SignalBuffer_t device_buffer, SignalBuffer_t tmp)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	cuComplex sample = get_sample(tmp, k);
	set_nr_sample(device_buffer, k, sample);
}


CUDAFFT::CUDAFFT(AbstractSignalProcessor* previous, size_t points) : CUDASignalProcessor(previous)
{
	this->points = points;
}

CUDAFFT::~CUDAFFT()
{
	cuda_deallocate_signal_buffer(this->tmp);
	CUDASignalProcessor::~CUDASignalProcessor();
}

void CUDAFFT::init(size_t max_buffer_size)
{
	//cudaError_t status;
	//status = cuda_allocate_signal_buffer(this->tmp, max_buffer_size);
	//check_cuda_status(status);
	CUDASignalProcessor::init(max_buffer_size);
}

void CUDAFFT::exec_kernel(SignalBuffer_t& host_buffer, SignalBuffer_t& device_buffer)
{
	cudaError_t status;

	if (get_buffer_size(host_buffer) <= 0)
		return;

	status = cuda_recreate_signal_buffer(device_buffer, points, 1);
	check_cuda_status(status);
	status = cuda_recreate_signal_buffer(tmp, points, 1);
	check_cuda_status(status);

	size_t buf_size = get_buffer_size(host_buffer);
	size_t butterflies = points / 2;
	size_t levels = (size_t)log2f(points);
	
	dim3 threadsPerBlock;
	dim3 blocks;
	get_threads_blocks_count(points, threadsPerBlock, blocks);

	set_buffer_size(tmp, points);
	
	cudafft_kernel_br_sort << <blocks, threadsPerBlock >> > (device_buffer, tmp, levels);
	get_threads_blocks_count(butterflies, threadsPerBlock, blocks);

	wait_for_gpu();

	for (size_t i = 0; i < levels; i++) {
		cudafft_kernel_butterflies<< <blocks, threadsPerBlock >> > (tmp, tmp, i);
		wait_for_gpu();
	}
	get_threads_blocks_count(points, threadsPerBlock, blocks);

	set_buffer_size(device_buffer, points);
	cudafft_kernel_copy << <blocks, threadsPerBlock >> > (device_buffer, tmp);

}