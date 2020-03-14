# include "SignalProcessor.h"

SignalProcessor::SignalProcessor(AbstractSignalProcessor* previous)
{
	this->previous = previous;
}

SignalProcessor::~SignalProcessor(){}

AbstractSignalProcessor* SignalProcessor::get_previous_processor()
{
	return this->previous;
}

int SignalProcessor::has_previous_processor()
{
	return this->previous != NULL;
}

void SignalProcessor::init(size_t max_buffer_size)
{
	if (has_previous_processor())
		return this->previous->init(max_buffer_size);
}

void SignalProcessor::process_buffer(SignalBuffer_t &buffer){}