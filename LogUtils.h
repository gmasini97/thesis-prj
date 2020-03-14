#pragma once

#include <cstdio>

//#define LOGUTILS_LOG_ENABLE

#ifdef LOGUTILS_LOG_ENABLE
	#define LOG(...) fprintf(stdout, __VA_ARGS__);
#endif
#ifndef LOGUTILS_LOG_ENABLE
	#define LOG(...) ;
#endif