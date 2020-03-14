#pragma once


#include <stdint.h>
#include "cuda_runtime.h"

typedef uint32_t BitMask;

const BitMask BIT_MASK_ALL = 0xFFFFFFFF;

__device__ __host__ static inline BitMask clear_bitmask()
{
	return 0;
}

__device__ __host__ static inline void set_bit(BitMask* mask, uint8_t index)
{
	*mask = *mask | (1 << index);
}

__device__ __host__ static inline void clear_bit(BitMask* mask, uint8_t index)
{
	*mask = *mask & ~(1 << index);
}

__device__ __host__ static inline int is_bit_set(BitMask mask, uint8_t index)
{
	return mask & (1 << index);
}