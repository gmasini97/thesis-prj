#include "CsvFileWriter.h"


CsvFileWriter::CsvFileWriter(AbstractSignalProcessor* p, std::string filename) : SignalProcessor(p)
{
	this->filename = filename;
}
CsvFileWriter::~CsvFileWriter()
{
	fclose(this->file);
}

void CsvFileWriter::init(size_t max_buffer_size)
{
	this->file = fopen(filename.c_str(), "w");
	if (file == NULL)
		throw std::runtime_error("cannot open output file");

	this->count = 0;

	fprintf(this->file, "SMP_N,Re,Im\n");

	return SignalProcessor::init(max_buffer_size);
}

void CsvFileWriter::process_buffer(SignalBuffer_t &buffer)
{
	if (has_previous_processor())
		get_previous_processor()->process_buffer(buffer);

	size_t size = get_buffer_size(buffer);
	for (size_t i = 0; i < size; i++)
	{
		cuComplex sample = get_sample(buffer, i);
		fprintf(this->file, "%lli,%f,%f\n", count, sample.x, sample.y);
		count++;
	}
}