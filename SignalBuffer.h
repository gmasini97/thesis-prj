#pragma once

#include <stdlib.h>
#include <cuComplex.h>
#include "BitMaskUtils.h"
#include <stdexcept>

struct SignalBuffer_t
{
	cuComplex* samples;
	size_t size;
	size_t max_size;
};
typedef struct SignalBuffer_t SignalBuffer_t;


SignalBuffer_t create_signal_buffer(size_t max_size);
void recreate_signal_buffer(SignalBuffer_t& in, size_t max_size, int copy_data);
void delete_signal_buffer(SignalBuffer_t &buffer);
void clear_signal_buffer(SignalBuffer_t &buffer);
void clear_signal_buffer_deep(SignalBuffer_t &buffer);
void copy_signal_buffer(SignalBuffer_t &dest, SignalBuffer_t src);

void signal_buffer_from_real_floats(SignalBuffer_t &buffer, float* values, size_t size);
size_t signal_buffer_to_real_floats(SignalBuffer_t& buffer, float* values, size_t max_size);



__host__ __device__ static inline SignalBuffer_t empty_signal_buffer()
{
	return {NULL, 0, 0};
}

// returns the value of the sample at a position; returns 0+i0 if index is out of size
__host__ __device__ static inline cuComplex get_sample(SignalBuffer_t buf, size_t index)
{
	if (index >= buf.size)
		return make_cuComplex(0, 0);
	return buf.samples[index];
}

__host__ __device__ static inline size_t get_max_buffer_size(SignalBuffer_t buf)
{
	return buf.max_size;
}

__host__ __device__ static inline size_t get_buffer_size(SignalBuffer_t buf)
{
	return buf.size;
}

__host__ __device__ static inline int set_buffer_size(SignalBuffer_t &buf, size_t size)
{
	if (size > buf.max_size)
		return 0;
	buf.size = size;
	return 1;
}

// sets the sample at a position, does nothing if index is out of bounds
__host__ __device__ static inline int set_nr_sample(SignalBuffer_t buf, size_t index, cuComplex v)
{
	if (index >= buf.max_size)
		return 0;
	buf.samples[index] = v;
	return 1;
}


// sets the sample at a position, tries to resize if index is out of bounds
__host__ __device__ static inline int set_sample(SignalBuffer_t &buf, size_t index, cuComplex v)
{
	if (!set_nr_sample(buf, index, v))
		return 0;
	if (index > buf.size)
		set_buffer_size(buf, index+1);
	return 1;
}