#include "Averager.h"

Averager::Averager(AbstractSignalProcessor* p) : SignalProcessor(p)
{
	this->count=0;
}

Averager::~Averager()
{
	delete_signal_buffer(avg);
}

void Averager::process_buffer(SignalBuffer_t& buffer)
{
	while (1)
	{
		if (has_previous_processor())
			get_previous_processor()->process_buffer(buffer);
		if (get_buffer_size(buffer) == 0)
			break;
		size_t max_size = get_max_buffer_size(buffer);
		recreate_signal_buffer(avg, max_size, 1);
		size_t size = get_buffer_size(buffer);
		cuComplex c0 = make_cuComplex(count, 0);
		cuComplex c1 = make_cuComplex(count+1, 0);
		for (size_t i = 0; i < size; i++)
		{
			cuComplex a = get_sample(avg, i);
			cuComplex x = get_sample(buffer, i);
			a = cuCmulf(a, c0);
			a = cuCaddf(a, x);
			a = cuCdivf(a, c1);
			set_sample(avg, i, a);
		}
		count++;
	}

	if (count == 0)
		return;

	size_t size = get_buffer_size(avg);
	for (size_t i = 0; i < size; i++)
	{
		cuComplex a = get_sample(avg, i);
		set_sample(buffer, i, a);
	}
	count = 0;
}