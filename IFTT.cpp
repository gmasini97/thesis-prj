
#include "IFTT.h"


void idft_wsio(SignalBuffer_t &bufferIn, SignalBuffer_t &bufferOut, size_t size)
{
	cuComplex* tmp = new cuComplex[size];
	cuComplex half_size = make_cuComplex(size/2, 0);
	cuComplex div_two = make_cuComplex(1 / 2, 0);
	cuComplex sample;
	for (size_t i = 0; i < size/2+1; i++)
	{
		sample = get_sample(bufferIn, i);
		sample = cuCdivf(sample, half_size);
		set_sample(bufferIn, i, cuConjf(sample));
		set_sample(bufferOut, i*2+1, make_cuComplex(0,0));
		set_sample(bufferOut, i * 2, make_cuComplex(0, 0));
	}

	sample = get_sample(bufferIn, 0);
	sample.x /= 2;
	set_sample(bufferIn, 0, sample);
	sample = get_sample(bufferIn, size/2+1);
	sample.x /= 2;
	set_sample(bufferIn, size / 2 + 1, sample);


	for (size_t k = 0; k < size; k++)
	{
		for (size_t i = 0; i < size / 2 + 1; i++) {
			cuComplex out = get_sample(bufferOut, i);
			sample = get_sample(bufferIn, k);
			cuComplex waves = cuComplex_exp(2 * M_PI * k * i / size);
			out.x += sample.x * waves.x;
			out.y += sample.y * waves.y;
			set_sample(bufferOut, i, out);
		}
	}

	delete[] tmp;

}

void ifft(SignalBuffer_t &buffer)
{
	size_t size = get_buffer_size(buffer);
	ifft_wsio(buffer, buffer, size);
}

void ifft_wsio(SignalBuffer_t &bufferIn, SignalBuffer_t &bufferOut, size_t size)
{
	cuComplex sample;

	for (size_t k = 0; k < size; k++)
	{
		sample = get_sample(bufferIn, k);
		sample = cuConjf(sample);
		set_sample(bufferOut, k, sample);
	}

	fft_wsio(bufferOut, bufferOut, size);

	cuComplex size_cmplx = make_cuComplex(size, 0);

	for (size_t i = 0; i < size; i++)
	{
		sample = get_sample(bufferOut, i);
		sample = cuConjf(cuCdivf(sample, size_cmplx));
		set_sample(bufferOut, i, sample);
	}
}

IFFTProcessor::IFFTProcessor(AbstractSignalProcessor* previous) : SignalProcessor(previous)
{}

void IFFTProcessor::process_buffer(SignalBuffer_t& buffer)
{
	if (has_previous_processor())
		get_previous_processor()->process_buffer(buffer);

	size_t size = get_buffer_size(buffer);
	if (size> 0)
		ifft(buffer);
}