#include "CmplxUtils.cuh"

__device__ __host__ cuComplex cuComplex_exp(float exp)
{
	float re = cos(exp);
	float im = sin(exp);
	return make_cuComplex(re, im);
}