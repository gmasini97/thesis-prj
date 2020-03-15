#include "Convolver.h"

#define SIGNAL_CHANNEL 0


size_t bounded_index(SignalBuffer_t buffer, size_t index)
{
	size_t channel_size = get_max_buffer_size(buffer);
	return index >= channel_size ? index % channel_size : index;
}





Convolver::Convolver(AbstractSignalProcessor* previous, SignalBuffer_t signal) : SignalProcessor(previous)
{
	this->signal = signal;
	this->temp_index = 0;
	this->samples_remaining = 0;
}

Convolver::~Convolver()
{
	if(has_previous_processor())
		delete get_previous_processor();
	delete_signal_buffer(this->temp);
}

void Convolver::process_buffer(SignalBuffer_t &buffer)
{
	if (has_previous_processor())
		get_previous_processor()->process_buffer(buffer);

	size_t buffer_size = get_buffer_size(buffer);
	size_t signal_size = get_buffer_size(signal);

	size_t total = buffer_size + signal_size - 1;
	recreate_signal_buffer(temp, total, 1);

	if ((buffer_size == 0 || signal_size == 0) && samples_remaining > 0)
	{
		size_t count = 0;
		cuComplex sample = get_sample(this->temp, temp_index);
		while (samples_remaining > 0 &&
				set_sample(buffer, count, sample)) {
			count++;
			temp_index = bounded_index(this->temp, temp_index + 1);
			sample = get_sample(this->temp, temp_index);
			samples_remaining--;
		}
		return;
	}

	
	for (size_t i = 0; i < buffer_size; i++)
	{
		cuComplex in_sample = get_sample(buffer, i);
		for (size_t j = 0; j < signal_size; j++)
		{
			cuComplex signal_sample = get_sample(signal, j);
			size_t index = bounded_index(this->temp, temp_index + i + j);
			cuComplex out_sample = get_sample(this->temp, index);
			cuComplex result = cuCaddf(out_sample, cuCmulf(in_sample, signal_sample));
			set_sample(this->temp, index, result);
		}
	}

	for (size_t i = 0; i < buffer_size; i++)
	{
		size_t index = bounded_index(this->temp, temp_index + i);
		cuComplex out_sample = get_sample(this->temp, index);
		set_sample(buffer, i, out_sample);
		set_sample(this->temp, index, make_cuComplex(0, 0));
	}

	this->temp_index = bounded_index(this->temp, temp_index + buffer_size);
	this->samples_remaining = signal_size - 1;

}








Convolver* create_convolver_from_file(AbstractSignalProcessor* previous, std::string filename, size_t conv_size)
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

	return new Convolver(previous, buffer);
}