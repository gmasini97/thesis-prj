#include "FFTConvolution.h"

#define SIGNAL_CHANNEL 0


size_t fft_bounded_index(SignalBuffer_t buffer, size_t channel, size_t index)
{
	size_t channel_size = get_max_possible_channel_buffer_size(buffer, channel);
	return index >= channel_size ? index % channel_size : index;
}


size_t get_fft_size(size_t a)
{
	size_t fft_size = (size_t)pow(2, ceil(log2(a)));
	return fft_size;
}






FFTConvolver::FFTConvolver(AbstractSignalProcessor* previous, BitMask channels_to_process, SignalBuffer_t signal) : SignalProcessor(previous, channels_to_process)
{
	this->signal = signal;
}

FFTConvolver::~FFTConvolver()
{
	if (has_previous_processor())
		delete get_previous_processor();
	delete_signal_buffer(this->fft_input);
	delete_signal_buffer(this->fft_signal);
	delete_signal_buffer(this->temp);
	delete[] this->temp_indexes;
	delete[] this->samples_remaining;
}

int FFTConvolver::init(size_t max_buffer_size, size_t channels)
{
	delete_signal_buffer(this->fft_input);
	delete_signal_buffer(this->fft_signal);
	delete_signal_buffer(this->temp);

	size_t extra_samples = (get_max_buffer_size(this->signal) - 1) * channels;
	size_t fft_size = get_fft_size(max_buffer_size + extra_samples);

	this->fft_input = create_signal_buffer(fft_size, channels);
	clear_signal_buffer_deep(this->fft_input);

	this->fft_signal = create_signal_buffer(fft_size, channels);
	clear_signal_buffer_deep(this->fft_signal);

	this->temp = create_signal_buffer(extra_samples, channels);
	clear_signal_buffer_deep(this->temp);

	this->temp_indexes = new size_t[channels]{ 0 };
	this->samples_remaining = new size_t[channels]{ 0 };

	return SignalProcessor::init(max_buffer_size, channels);
}

void FFTConvolver::process_buffer(SignalBuffer_t* buffer)
{

	if (has_previous_processor())
		get_previous_processor()->process_buffer(buffer);
	size_t channels = get_channels(*buffer);

	for (size_t channel = 0; channel < channels; channel++)
	{
		if (!has_to_process_channel(channel))
			continue;

		size_t input_size = get_channel_buffer_size(*buffer, channel);
		size_t signal_size = get_channel_buffer_size(signal, SIGNAL_CHANNEL);
		size_t out_size = input_size + signal_size - 1;
		size_t fft_size = get_fft_size(out_size);

		if (input_size == 0 || signal_size == 0)
		{
			size_t count = 0;
			cuComplex x = get_signal_buffer_sample(temp, channel, temp_indexes[channel]);
			
			while (samples_remaining[channel] > 0 &&
					set_signal_buffer_sample(*buffer, channel, count, x)) {
				samples_remaining[channel] --;
				count++;
				temp_indexes[channel] = fft_bounded_index(temp, channel, temp_indexes[channel] + 1);
				x = get_signal_buffer_sample(temp, channel, temp_indexes[channel]);
			}
			continue;
		}

		size_t temp_index = temp_indexes[channel];

		clear_signal_buffer(fft_signal);
		clear_signal_buffer(fft_input);

		LOG("sizes, in: %lli, cnv: %lli, out: %lli, fft: %lli\n", input_size, signal_size, out_size, fft_size);

		fft_wsio(buffer, &fft_input, channel, fft_size);
		fft_wsio(&signal, &fft_signal, SIGNAL_CHANNEL, fft_size);
		//dft_wsio(buffer, &fft_input, channel, fft_size);
		//dft_wsio(&signal, &fft_signal, SIGNAL_CHANNEL, fft_size);

		for (size_t i = 0; i < fft_size; i++)
		{
			cuComplex a = get_signal_buffer_sample(fft_input, channel, i);
			cuComplex b = get_signal_buffer_sample(fft_signal, SIGNAL_CHANNEL, i);
			cuComplex y = cuCmulf(a, b);
			set_signal_buffer_sample(fft_input, channel, i, y);
		}

		ifft_wsio(&fft_input, &fft_input, channel, fft_size);
		LOG("512th: %.3f\n", get_signal_buffer_sample(fft_input, channel, 512).x);

		for (size_t i = 0; i < input_size; i++)
		{
			size_t index = fft_bounded_index(temp, channel, temp_index + i);
			cuComplex a = get_signal_buffer_sample(temp, channel, index);
			cuComplex b = get_signal_buffer_sample(fft_input, channel, i);
			cuComplex y = cuCaddf(a, b);
			set_signal_buffer_sample(*buffer, channel, i, b);
			set_signal_buffer_sample(temp, channel, index, make_cuComplex(0,0));
		}

		for (size_t i = input_size; i < out_size; i++)
		{
			size_t index = fft_bounded_index(temp, channel, temp_index + i);
			cuComplex x = get_signal_buffer_sample(fft_input, channel, i);
			set_signal_buffer_sample(temp, channel, index, x);
		}

		temp_indexes[channel] += input_size;
		samples_remaining[channel] = signal_size - 1;
	}

}







FFTConvolver* create_fftconvolver_from_file(AbstractSignalProcessor* previous, std::string filename, size_t conv_size)
{
	SF_INFO info;
	memset(&info, 0, sizeof(SF_INFO));
	SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &info);

	if (info.channels != 1) {
		std::cout << "only 1 channel convolution kernel allowed" << std::endl;
		return NULL;
	}

	float* real = new float[conv_size];
	float* imag = new float[conv_size] {0};

	size_t actual_read = sf_read_float(file, real, conv_size);

	SignalBuffer_t buffer = create_signal_buffer(conv_size, 1);
	signal_buffer_from_floats(buffer, real, imag, actual_read);

	sf_close(file);
	delete[] real;
	delete[] imag;

	return new FFTConvolver(previous, mask, buffer);
}