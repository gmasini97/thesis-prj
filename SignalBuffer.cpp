#include "SignalBuffer.h"

void recreate_signal_buffer(SignalBuffer_t &in, size_t max_size, int copy_data)
{
	size_t size = get_max_buffer_size(in);
	if (max_size > size) {
		size_t old_size = get_buffer_size(in);
		SignalBuffer_t buffer = create_signal_buffer(max_size);
		if (copy_data)
			copy_signal_buffer(buffer, in);
		delete_signal_buffer(in);
		set_buffer_size(buffer, old_size);
		in = buffer;
	}
}

SignalBuffer_t create_signal_buffer(size_t max_size)
{
	SignalBuffer_t buffer;
	buffer.max_size = max_size;
	if (cudaMallocHost((void**)&(buffer.samples), sizeof(cuComplex) * max_size) != cudaSuccess)
		throw std::runtime_error("cuda malloc host fail");
	buffer.size = 0;
	return buffer;
}

void copy_signal_buffer(SignalBuffer_t& dest, SignalBuffer_t src)
{
	size_t dest_size = get_max_buffer_size(dest);
	size_t src_size = get_buffer_size(src);

	for (size_t i = 0; i < dest_size; i++)
	{
		cuComplex sample;
		if (i < src_size)
			sample = get_sample(src, i);
		else
			sample = make_cuComplex(0,0);
		set_sample(dest, i, sample);
	}
}

void delete_signal_buffer(SignalBuffer_t &buffer)
{
	if (buffer.samples != NULL) {
		if (cudaFreeHost(buffer.samples) != cudaSuccess)
			throw std::runtime_error("cuda free host fail");
	}
	buffer.size = 0;
	buffer.max_size = 0;
}

void clear_signal_buffer(SignalBuffer_t &buffer)
{
	buffer.size = 0;
}

void clear_signal_buffer_deep(SignalBuffer_t &buffer)
{
	for (size_t i = 0; i < buffer.max_size; i++)
	{
		buffer.samples[i] = make_cuComplex(0,0);
	}
	clear_signal_buffer(buffer);
}

void signal_buffer_from_real_floats(SignalBuffer_t &buffer, float* values, size_t size)
{
	clear_signal_buffer(buffer);
	for (size_t i = 0; i < size; i++)
	{
		cuComplex v = make_cuComplex(values[i], 0);
		set_sample(buffer, i, v);
	}
}

size_t signal_buffer_to_real_floats(SignalBuffer_t& buffer, float* values, size_t max_size)
{
	size_t buf_size = get_buffer_size(buffer);
	size_t size = buf_size < max_size ? buf_size : max_size;

	for (size_t i = 0; i < size; i++)
	{
		cuComplex v = get_sample(buffer, i);
		values[i] = v.x;
	}

	return size;
}