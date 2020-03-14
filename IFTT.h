#pragma once

#include "SignalProcessor.h"
#include "FFT.h"

void ifft(SignalBuffer_t &buffer);
void ifft_wsio(SignalBuffer_t &b_in, SignalBuffer_t &b_out, size_t size);
void idft_wsio(SignalBuffer_t &b_in, SignalBuffer_t &b_out, size_t size);

class IFFTProcessor : public SignalProcessor
{
public:
	IFFTProcessor(AbstractSignalProcessor* previous);
	void process_buffer(SignalBuffer_t& buffer);
};