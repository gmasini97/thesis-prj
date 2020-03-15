#include "FFT.h"

void bit_reversal_sort_wsio(SignalBuffer_t &bufferIn, SignalBuffer_t &bufferOut, size_t size)
{
	cuComplex sample, temporary;
	size_t buffer_size = get_buffer_size(bufferIn);
	size_t j, k, halfSize;

	halfSize = size / 2;
	j = halfSize;

	sample = get_sample(bufferIn, 0);
	set_sample(bufferOut, 0, sample);

	sample = get_sample(bufferIn, size-1);
	set_sample(bufferOut, size - 1, sample);

	for (size_t i = 1; i < size - 2; i++)
	{
		if (i < j)
		{
			temporary = get_sample(bufferIn, j);
			sample = get_sample(bufferIn, i);

			if (i >= buffer_size)
				sample = make_cuComplex(0,0);
			if (j >= buffer_size)
				temporary = make_cuComplex(0, 0);

			set_sample(bufferOut, j, sample);
			set_sample(bufferOut, i, temporary);
		}
		else if (i == j) {
			sample = get_sample(bufferIn, i);
			if (i >= buffer_size)
				sample = make_cuComplex(0, 0);
			set_sample(bufferOut, i, sample);
		}
		k = halfSize;
		while (k <= j)
		{
			j = j - k;
			k = k / 2;
		}
		j = j + k;
	}
}

void bit_reversal_sort(SignalBuffer_t &buffer)
{
	size_t size = get_buffer_size(buffer);
	bit_reversal_sort_wsio(buffer, buffer, size);
}

void butterfly_calculation(cuComplex* a, cuComplex* b, cuComplex w)
{
	cuComplex aa = *a;
	cuComplex bw = cuCmulf(*b, w);

	*a = cuCaddf(aa, bw);
	*b = cuCsubf(aa, bw);
}

void fft(SignalBuffer_t &buffer)
{
	size_t size = get_buffer_size(buffer);
	fft_wsio(buffer, buffer, size);
}

void fft_wsio(SignalBuffer_t &bufferIn, SignalBuffer_t &bufferOut, size_t size_in)
{

	cuComplex w, wm;

	size_t levels;
	size_t index_a, index_b;

	size_t size = (size_t)pow(2, ceil(log2(size_in)));;

	levels = (size_t)log2(size);

	bit_reversal_sort_wsio(bufferIn, bufferOut, size);

	for (size_t level = 0; level < levels; level++)
	{
		size_t butterflies_per_dft = (size_t)pow(2, level);
		size_t dfts = size / (butterflies_per_dft * 2);

		wm = cuComplex_exp(-(M_PI / butterflies_per_dft));
		w = make_cuComplex(1,0);
		for (size_t butterfly = 0; butterfly < butterflies_per_dft; butterfly++)
		{
			for (size_t dft = 0; dft < dfts; dft++)
			{
				index_a = butterfly + dft * (butterflies_per_dft * 2);
				index_b = index_a + butterflies_per_dft;
				cuComplex a = get_sample(bufferOut, index_a);
				cuComplex b = get_sample(bufferOut, index_b);
				butterfly_calculation(&a, &b, w);
				set_nr_sample(bufferOut, index_a, a);
				set_nr_sample(bufferOut, index_b, b);
			}
			w = cuCmulf(w, wm);
		}
		set_buffer_size(bufferOut, get_buffer_size(bufferIn));
	}
}

FFTProcessor::FFTProcessor(AbstractSignalProcessor* previous, size_t points) : SignalProcessor(previous)
{
	this->points = points;
}

void FFTProcessor::process_buffer(SignalBuffer_t &buffer)
{
	if (has_previous_processor())
		get_previous_processor()->process_buffer(buffer);
	size_t size = get_buffer_size(buffer);
	recreate_signal_buffer(buffer, points, 1);
	if (size > 0) {
		fft_wsio(buffer, buffer, points);
	}
}