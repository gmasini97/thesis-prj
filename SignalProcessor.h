#pragma once

#include "SignalBuffer.h"
#include "BitMaskUtils.h"
#include "LogUtils.h"
#include <stdexcept>

#define PI 3.141592654f
#define M_PI 3.141592654f

class AbstractSignalProcessor
{
public:
	virtual AbstractSignalProcessor* get_previous_processor() = 0;
	virtual void init(size_t max_buffer_size) = 0;
	virtual void process_buffer(SignalBuffer_t &buffer) = 0;
};


class SignalProcessor : public AbstractSignalProcessor
{
private:
	AbstractSignalProcessor* previous;
public:
	SignalProcessor(AbstractSignalProcessor* previous);
	~SignalProcessor();

	AbstractSignalProcessor* get_previous_processor();
	int has_previous_processor();

	virtual void init(size_t max_buffer_size);
	virtual void process_buffer(SignalBuffer_t &buffer);
};