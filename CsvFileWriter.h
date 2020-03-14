#pragma once

#define _CRT_SECURE_NO_WARNINGS
#include "SignalProcessor.h"
#include <iostream>

class CsvFileWriter : public SignalProcessor
{
private:
	std::string filename;
	FILE* file;
	size_t count;
public:
	CsvFileWriter(AbstractSignalProcessor* p, std::string filename);
	~CsvFileWriter();
	void init(size_t max_buffer_size);
	void process_buffer(SignalBuffer_t &buffer);
};