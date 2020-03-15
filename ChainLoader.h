#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include "SignalProcessor.h"
#include "CsvFileWriter.h"
#include "Convolver.h"
//#include "FFTConvolution.h"
#include "SndFileWriter.h"
#include "DFT.h"
#include "CUDADFT.cuh"
#include "CUDAConvolver.cuh"
#include "CUDAFFT.cuh"
#include "Averager.h"
#include "FFT.h"
#include "IFTT.h"
#include <sstream>
#include <fstream>

#define MAX_LINE_SIZE 256

AbstractSignalProcessor* build_fx_chain(std::string filename, AbstractSignalProcessor* previous);
AbstractSignalProcessor* build_fx_chain_rec(std::ifstream* file, AbstractSignalProcessor* previous);
AbstractSignalProcessor* create_from_line(std::string line, AbstractSignalProcessor* previous);