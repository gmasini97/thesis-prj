#pragma once
#include "SignalProcessor.h"
#include "CmplxUtils.cuh"

void dft(SignalBuffer_t &buffer);
void dft_wsio(SignalBuffer_t &bufferIn, SignalBuffer_t &bufferOut, size_t size);

class DFTProcessor : public SignalProcessor
{
private:
	size_t points;
public:
	DFTProcessor(AbstractSignalProcessor* p, size_t points);
	~DFTProcessor();
	void process_buffer(SignalBuffer_t &buffer);
};