#include "StdAfx.h"

void Delay(double ms)
{
	LARGE_INTEGER s, e, f;
	double freq;

	// Grab the frequency.
	QueryPerformanceFrequency(&f);
	freq = double(f.QuadPart);

	QueryPerformanceCounter(&e);
	s = e;

	while ((double)(e.QuadPart - s.QuadPart)/freq*1000.0 < ms) {
		QueryPerformanceCounter(&e);
	}
}
