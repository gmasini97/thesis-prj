#pragma once

#include "SignalProcessor.h"
#include <sndfile.h>
#include <stdlib.h>
#include <string>

class SndFileWriter : public SignalProcessor
{
private:
	SNDFILE* file;
	std::string filename;
	SF_INFO info;

	size_t max_reals_size;
	float* reals;
public:
	SndFileWriter(AbstractSignalProcessor* p, std::string filename, SF_INFO info);
	~SndFileWriter();
	SF_INFO get_info();
	void init(size_t max_buffer_size);
	void process_buffer(SignalBuffer_t &buffer);
};