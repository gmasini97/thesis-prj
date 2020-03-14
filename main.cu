#include <iostream>
#include "CsvFileWriter.h"
#include "SndFileLoader.h"
#include "ChainLoader.h"

int main(int argc, const char** argv)
{
	if (argc < 4)
		std::cout << "not enough args" << std::endl;

	size_t buffer_size = atoi(argv[1]);

	SndFileLoader* fileLoader = new SndFileLoader(NULL, argv[3]);
	AbstractSignalProcessor* chain = build_fx_chain(argv[2], fileLoader);

	SignalBuffer_t buffer = create_signal_buffer(buffer_size);
	try
	{
		chain->init(buffer_size);
	}
	catch (int x)
	{
		LOG("err init");
		return 1;
	}

	auto start = std::chrono::high_resolution_clock::now();
	do
	{
		chain->process_buffer(buffer);
	} while(get_buffer_size(buffer) > 0);
	auto stop = std::chrono::high_resolution_clock::now();

	long long int time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

	std::cout << time << " us" << std::endl;

	return 0;
}
