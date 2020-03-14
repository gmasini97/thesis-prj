#pragma once

#include "SignalProcessor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "sndfile.h"


class Convolver : public SignalProcessor
{
private:
	SignalBuffer_t signal;
	SignalBuffer_t temp = empty_signal_buffer();

	size_t temp_index;
	size_t samples_remaining;
public:
	Convolver(AbstractSignalProcessor* previous, SignalBuffer_t signal);
	~Convolver();

	void process_buffer(SignalBuffer_t &buffer);
};

Convolver* create_convolver_from_file(AbstractSignalProcessor* previous, std::string filename, size_t conv_size);