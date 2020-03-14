#include "SndFileWriter.h"

SndFileWriter::SndFileWriter(AbstractSignalProcessor* p, std::string filename, SF_INFO info) : SignalProcessor(p)
{
	this->filename = filename;
	info.channels = 1;
	this->info = info;
	this->reals = NULL;
	this->max_reals_size = 0;
}

SndFileWriter::~SndFileWriter()
{
	sf_close(this->file);
	delete[] this->reals;
}

SF_INFO SndFileWriter::get_info()
{
	return this->info;
}

void SndFileWriter::init(size_t max_buffer_size)
{
	this->file = sf_open(this->filename.c_str(), SFM_WRITE, &(this->info));
	if (this->file == NULL)
		throw std::runtime_error("cannot open output snd file");

	return SignalProcessor::init(max_buffer_size);
}


void SndFileWriter::process_buffer(SignalBuffer_t &buffer)
{
	if (has_previous_processor())
		get_previous_processor()->process_buffer(buffer);

	size_t max_size = get_max_buffer_size(buffer);
	if (reals == NULL || max_reals_size < max_size) {
		if (reals != NULL)
			delete[] reals;
		reals = new float[max_size];
		max_reals_size = max_size;
	}

	size_t size = signal_buffer_to_real_floats(buffer, reals, max_reals_size);

	sf_write_float(this->file, reals, size);
}