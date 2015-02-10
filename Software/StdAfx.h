/************************************************************************
	StdAfx.h -- Precompiled header
************************************************************************/

#pragma once

// wxWindows
#include <wx\wxprec.h>
#include <wx\glcanvas.h>

// C++ libs
#include <string>
#include <sstream>
#include <vector>
#include <hash_map>
#include <queue>
#include <math.h>

// OpenGL
#include <gl\glu.h>
#include <gl\glut.h>

// ComputerBoards
#include <cbw2005.h>

// Tempo -- must use a C-style include
extern "C"	{
#include <Rdx.h>
}

#include <MatlabRDX.h>

void Delay(double ms);