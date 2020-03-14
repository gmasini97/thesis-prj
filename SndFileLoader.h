#pragma once

#include "SignalProcessor.h"
#include <sndfile.h>
#include <stdlib.h>

class SndFileLoader : public SignalProcessor
{
private:
	SNDFILE* file;
	const char* filename;
	SF_INFO info;

	float* reals;
	size_t reals_size;
public:
	SndFileLoader(AbstractSignalProcessor* p, const char* filename);
	~SndFileLoader();
	SF_INFO get_info();
	void init(size_t max_buffer_size);
	void process_buffer(SignalBuffer_t &buffer);
};