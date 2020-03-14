#pragma once

#include "SignalProcessor.h"
#include "FFT.h"
#include "DFT.h"
#include "IFTT.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "sndfile.h"


class FFTConvolver : public SignalProcessor
{
private:
	SignalBuffer_t signal;
	SignalBuffer_t fft_input = empty_signal_buffer();
	SignalBuffer_t fft_signal = empty_signal_buffer();
	SignalBuffer_t temp = empty_signal_buffer();

	size_t* temp_indexes;
	size_t* samples_remaining;
public:
	FFTConvolver(AbstractSignalProcessor* previous, BitMask channels_to_process, SignalBuffer_t signal);
	~FFTConvolver();

	void init(size_t max_buffer_size);
	void process_buffer(SignalBuffer_t &buffer);
};

FFTConvolver* create_fftconvolver_from_file(AbstractSignalProcessor* previous, std::string filename, size_t conv_size);