#include "DFT.h"

void dft_wsio(SignalBuffer_t &bufferIn, SignalBuffer_t &bufferOut, size_t size)
{
	cuComplex* tmp = new cuComplex[size];
	cuComplex sample, s;

	for (size_t k = 0; k < size; k++)
	{
		tmp[k] = make_cuFloatComplex(0,0);
		for (size_t i = 0; i < size; i++)
		{
			s = cuComplex_exp(-2 * M_PI * k * i / size);
			sample = get_sample(bufferIn, i);

			tmp[k] = cuCaddf(tmp[k], cuCmulf(sample, s));
		}
	}

	for (size_t k = 0; k < size; k++)
	{
		set_sample(bufferOut, k, tmp[k]);
	}
	delete[] tmp;
}

void dft(SignalBuffer_t &buffer)
{
	size_t size = get_max_buffer_size(buffer);
	dft_wsio(buffer, buffer, size);
}

DFTProcessor::DFTProcessor(AbstractSignalProcessor* p, size_t points) : SignalProcessor(p)
{
	this->points=points;
}

DFTProcessor::~DFTProcessor()
{
}


void DFTProcessor::process_buffer(SignalBuffer_t &buffer)
{
	if (has_previous_processor()) {
		get_previous_processor()->process_buffer(buffer);

		recreate_signal_buffer(buffer, points, 1);
		if (get_buffer_size(buffer) > 0)
			dft(buffer);
	}
}