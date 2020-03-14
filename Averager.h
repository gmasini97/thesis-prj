#pragma once
#include "SignalProcessor.h"
#include "CmplxUtils.cuh"

class Averager : public SignalProcessor
{
private:
	size_t count;
	SignalBuffer_t avg = empty_signal_buffer();
public:
	Averager(AbstractSignalProcessor* p);
	~Averager();
	void process_buffer(SignalBuffer_t& buffer);
};