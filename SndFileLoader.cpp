#include "SndFileLoader.h"


SndFileLoader::SndFileLoader(AbstractSignalProcessor* p, const char* filename) : SignalProcessor(p)
{
	this->filename = filename;
	this->reals = NULL;
	this->reals_size = 0;
}

SndFileLoader::~SndFileLoader()
{
	sf_close(this->file);
	delete[] this->reals;
}

SF_INFO SndFileLoader::get_info()
{
	return this->info;
}

void SndFileLoader::init(size_t max_buffer_size)
{
	this->file = sf_open(this->filename, SFM_READ, &(this->info));
	if (this->file == NULL)
		throw std::runtime_error("cannot open input file");

	if (info.channels > 1)
		throw std::invalid_argument("input file must be of 1 channel");

	reals = new float[max_buffer_size];
	reals_size = max_buffer_size;
	SignalProcessor::init(max_buffer_size);
}


void SndFileLoader::process_buffer(SignalBuffer_t &buffer)
{
	if (has_previous_processor())
		get_previous_processor()->process_buffer(buffer);
	
	size_t to_read = reals_size;

	size_t read_count = sf_read_float(this->file, reals, to_read);
	
	signal_buffer_from_real_floats(buffer, reals, read_count);
}