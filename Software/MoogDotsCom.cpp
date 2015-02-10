#include "StdAfx.h"
#include "MoogDotsCom.h"
#include "GLWindow.h"
#include <wx/busyinfo.h>
#include "MoogDots.h"
#include <process.h>    /* _beginthread, _endthread */


// function pointer typdefs
typedef void (APIENTRY *PFNWGLEXTSWAPCONTROLPROC) (int);
typedef int (*PFNWGLEXTGETSWAPINTERVALPROC) (void);

// declare functions
PFNWGLEXTSWAPCONTROLPROC wglSwapIntervalEXT = NULL;
PFNWGLEXTGETSWAPINTERVALPROC wglGetSwapIntervalEXT = NULL;

// Parameter list -- Original declaration can be found in ParameterList.cpp
extern CParameterList g_pList;
extern float CENTER2SCREEN;
extern int SCREEN_WIDTH;
extern int SCREEN_HEIGHT;


#include <cstdlib>
#include <sys/timeb.h>

double getCurrentTimeInSec(){

	timeb tb;
	ftime(&tb);
	int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
	return double(nCount);
}

MoogDotsCom::MoogDotsCom(CMainFrame *mainFrame, char *mbcIP, int mbcPort, char *localIP, int localPort, bool useCustomTimer) :
			 CORE_CONSTRUCTOR, m_glWindowExists(false), m_isLibLoaded(false), m_messageConsole(NULL),
		     m_tempoHandle(-1), m_drawRegularFeedback(true),
			/* m_previousLateral(0.0), m_previousSurge(0.0), m_previousHeave(MOTION_BASE_CENTER), */
			 m_previousBitLow(true),
			 stimAnalogOutput(0),stimAnalogMult(1.0), openGLsignal(0.0), motionType(0)
{
	m_mainFrame = mainFrame;

	//WinExec("C:\\usr\\local\\bin\\MouseVR.exe",SW_SHOWNORMAL); 

	// Create the OpenGL display window.
#if DUAL_MONITORS
#if FLIP_MONITORS
	m_glWindow = new GLWindow("GL Window", SCREEN_WIDTH, 0, SCREEN_WIDTH, SCREEN_HEIGHT, createFrustum(), createStarField());
#else
	m_glWindow = new GLWindow("GL Window", 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, createFrustum(), createStarField());
#endif
#else
	m_glWindow = new GLWindow("GL Window", 450, 100, 800, 870, createFrustum(), createStarField());
#endif

#if SHOW_GL_WINDOW
	m_glWindow->Show(true);
#endif

//#if CUSTOM_TIMER - Johnny 6/17/2007
	m_doSyncPulse = false;
	m_customTimer = false;
//#endif;

	// Initialize the previous position data.
	m_previousPosition.heave = MOTION_BASE_CENTER; m_previousPosition.lateral = 0.0f;
	m_previousPosition.surge = 0.0f; m_previousPosition.roll = 0.0f;
	m_previousPosition.yaw = 0.0f; m_previousPosition.pitch = 0.0f;

	m_glWindowExists = true;
	m_setRotation = false;
	m_continuousMode = false;

#if VERBOSE_MODE
	m_verboseMode = true;
#else
	m_verboseMode = false;
#endif

#if LISTEN_MODE
	m_listenMode = true;
#else
	m_listenMode = false;
#endif

	QueryPerformanceFrequency(&m_freq);

	m_delay = 0.0;

	// Set the packet rate.
	SetPacketRate(16.594);
	//SetPacketRate(16.594/2.0);

#if MINI_MOOG_SYSTEM
	Create(mbcIP, mbcPort, localIP, localPort);
#endif

//#if !CUSTOM_TIMER - Johnny 6-12-07
	// Setup the vsync functions and, by default, turn off vsync.
	InitVSync();
	SetVSyncState(false);
//#endif - Johnny 6-12-07

	// Set the connection type to the default one.
#if DEF_LISTEN_MODE == 1
	m_connectionType = ConnectionType::Tempo;
#elif DEF_LISTEN_MODE == 2
	m_connectionType = ConnectionType::Pipes;
#else
	m_connectionType = ConnectionType::None;
#endif

	m_matlabRDX = NULL;

	m_pipeHandle = INVALID_HANDLE_VALUE;

	controlCounter = 0;
}

double const MoogDotsCom::m_speedBuffer[SPEED_BUFFER_SIZE] =
{
	EXP_BASE, pow(EXP_BASE, 2), pow(EXP_BASE, 3), pow(EXP_BASE, 4), pow(EXP_BASE, 5),
	pow(EXP_BASE, 6), pow(EXP_BASE, 7), pow(EXP_BASE, 8), pow(EXP_BASE, 9), pow(EXP_BASE, 10),
	pow(EXP_BASE, 11), pow(EXP_BASE, 12), pow(EXP_BASE, 13), pow(EXP_BASE, 14), pow(EXP_BASE, 15),
	pow(EXP_BASE, 16), pow(EXP_BASE, 17), pow(EXP_BASE, 18), pow(EXP_BASE, 19), pow(EXP_BASE, 20),
	pow(EXP_BASE, 21), pow(EXP_BASE, 22), pow(EXP_BASE, 23), pow(EXP_BASE, 24), pow(EXP_BASE, 25),
	pow(EXP_BASE, 26), pow(EXP_BASE, 27), pow(EXP_BASE, 28), pow(EXP_BASE, 29), pow(EXP_BASE, 30)
};

double const MoogDotsCom::m_speedBuffer2[SPEED_BUFFER_SIZE] =
{
	0.9988, 0.9966, 0.9930, 0.9872, 0.9782, 0.9649, 0.9459, 0.9197,
	0.8852, 0.8415, 0.7882, 0.7257, 0.6554, 0.5792, 0.5000, 0.4208,
	0.3446, 0.2743, 0.2118, 0.1585, 0.1148, 0.0803, 0.0541, 0.0351,
	0.0218, 0.0128, 0.0070, 0.0034, 0.0012, 0.0
};

MoogDotsCom::~MoogDotsCom()
{
	if (m_glWindowExists) { 
		m_glWindow->Destroy();
	}

#if USE_MATLAB_RDX
	if (m_matlabRDX != NULL) {
		delete m_matlabRDX;
	}
#endif
}


//#if !CUSTOM_TIMER - Johnny 6/17/07
void MoogDotsCom::InitVSync()
{
	//get extensions of graphics card
	char* extensions = (char*)glGetString(GL_EXTENSIONS);

	// Is WGL_EXT_swap_control in the string? VSync switch possible?
	if (strstr(extensions,"WGL_EXT_swap_control"))
	{
		//get address's of both functions and save them
		wglSwapIntervalEXT = (PFNWGLEXTSWAPCONTROLPROC)
			wglGetProcAddress("wglSwapIntervalEXT");
		wglGetSwapIntervalEXT = (PFNWGLEXTGETSWAPINTERVALPROC)
			wglGetProcAddress("wglGetSwapIntervalEXT");
	}
}


void MoogDotsCom::SetVSyncState(bool enable)
{
	if (enable) {
       wglSwapIntervalEXT(1);
	}
	else {
       wglSwapIntervalEXT(0);
	}

	m_customTimer = enable;
}


bool MoogDotsCom::VSyncEnabled()
{
	return (wglGetSwapIntervalEXT() > 0);
}
//#endif - Johnny 6-12-07


void MoogDotsCom::ListenMode(bool value)
{
	m_listenMode = value;
}


void MoogDotsCom::ShowGLWindow(bool value)
{
	if (m_glWindowExists) {
		m_glWindow->Show(value);
	}
}


void MoogDotsCom::SetConnectionType(MoogDotsCom::ConnectionType ctype)
{
	m_connectionType = ctype;
}


MoogDotsCom::ConnectionType MoogDotsCom::GetConnectionType()
{
	return m_connectionType;
}


void MoogDotsCom::InitPipes()
{
	m_iocomplete = true;

	// Setup the security descriptor for the pipe to allow access from the network.
	InitializeSecurityDescriptor(&m_pipeSD, SECURITY_DESCRIPTOR_REVISION);
	SetSecurityDescriptorDacl(&m_pipeSD, TRUE, (PACL)NULL, FALSE);
	m_pipeSA.nLength = sizeof(m_pipeSA);
	m_pipeSA.lpSecurityDescriptor = &m_pipeSD;
	m_pipeSA.bInheritHandle = false;

	// Initialize the overlapped data structure.
	m_overlappedEvent.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
	m_overlappedEvent.Offset = (DWORD)0;
	m_overlappedEvent.OffsetHigh = (DWORD)0;
	if (m_overlappedEvent.hEvent == NULL) {
		wxMessageDialog d(NULL, "Failed to create overlapped event.  Pipes won't work");
		d.ShowModal();
	}

	// Initialize the pipe.
	m_pipeHandle = CreateNamedPipe("\\\\.\\pipe\\moogpipe", PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
							   PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
							   (DWORD)2, (DWORD)PIPE_BUFSIZE, (DWORD)PIPE_BUFSIZE, 500,
							   (LPSECURITY_ATTRIBUTES)&m_pipeSA);
	if (m_pipeHandle == INVALID_HANDLE_VALUE) {
		wxMessageDialog d(NULL, "Failed to create pipe.  Pipes won't work");
		d.ShowModal();
	}
}


void MoogDotsCom::ClosePipes()
{
	CloseHandle(m_overlappedEvent.hEvent);
	CloseHandle(m_pipeHandle);
	m_messageConsole->InsertItems(1, &wxString("Closed pipes connection."), 0);
}


void MoogDotsCom::InitTempo()
{
	int errCode;
	vector<wxString> errorStrings;

#if PCI_DIO_24H_PRESENT
	m_PCI_DIO24_Object.DIO_board_num = m_PCI_DIO24_Object.GetBoardNum("pci-dio24h");
	
#endif

#if USE_ANALOG_OUT_BOARD
	m_PCI_DIO48H_Object.DIO_board_num = m_PCI_DIO24_Object.GetBoardNum("pci-dda02/16");
#else
	m_PCI_DIO48H_Object.DIO_board_num = m_PCI_DIO24_Object.GetBoardNum("pci-dio48h");
#endif

#if JOHNNY_WORKSTATION
	//m_PCI_DIO24_Object.DIO_board_num = m_PCI_DIO24_Object.GetBoardNum("pci-dio24/s");
	m_PCI_DIO48H_Object.DIO_board_num = m_PCI_DIO24_Object.GetBoardNum("pci-dda08/16");
#endif
	m_PCI_DIO48H_Object.DIO_base_address = m_PCI_DIO48H_Object.Get8255BaseAddr(m_PCI_DIO48H_Object.DIO_board_num, 1) + 4;

	//RDX is handled either via chip #1 on the pci-dio48h (rig #3) or via
	//an ISA version of the DIO24 board (base address set to 0x300
	//The logic here makes this code portable across different configurations
	if (m_PCI_DIO48H_Object.DIO_board_num == -1) { //board not found
		m_RDX_base_address = 0x300;  //default for the ISA board
	}
	else {
		m_RDX_base_address = m_PCI_DIO48H_Object.DIO_base_address;
	}

#if PCI_DIO_24H_PRESENT
	//Configure the PCI DIO board for digital input on the first port and output on second
	errCode = cbDConfigPort(m_PCI_DIO24_Object.DIO_board_num, FIRSTPORTA, DIGITALIN);
	if (errCode != 0) {
		errorStrings.push_back("Error setting up PCI-DIO24 port A!");
	}
	errCode = cbDConfigPort(m_PCI_DIO24_Object.DIO_board_num, FIRSTPORTB, DIGITALOUT);
	if (errCode != 0) {
		errorStrings.push_back("Error setting up PCI-DIO24 port B!");
	}
#else
	// Configure the 48H board for ouput on the first chip, port B.
	errCode = cbDConfigPort(m_PCI_DIO48H_Object.DIO_board_num, FIRSTPORTB, DIGITALOUT);
	if (errCode != 0) {
		errorStrings.push_back("Error setting up PCI-DIO48H port B!");
	}

	// Configure the 48H board for input on the first chip, port A.
	errCode = cbDConfigPort(m_PCI_DIO48H_Object.DIO_board_num, FIRSTPORTA, DIGITALIN);
	if (errCode != 0) {
		errorStrings.push_back("Error setting up PCI-DIO48H port A!");
	}
#endif

#if USE_ANALOG_OUT_BOARD
	// Zero out PCI-DDA02/16 board.
	cbAOut(m_PCI_DIO48H_Object.DIO_board_num, 0, BIP10VOLTS, DASCALE(0));
	cbAOut(m_PCI_DIO48H_Object.DIO_board_num, 1, BIP10VOLTS, DASCALE(0));
#endif

	// Open up a Tempo handle.
	OpenTempo();
	
	// Print info out to the message console.
	if (m_messageConsole != NULL) {
		m_messageConsole->Append(wxString::Format("Tempo Handle: %d", m_tempoHandle));
		m_messageConsole->Append("--------------------------------------------------------------------------------");
#if PCI_DIO_24H_PRESENT
		m_messageConsole->Append(wxString::Format("PCI_DIO24 board # = %d", m_PCI_DIO24_Object.DIO_board_num));
#endif
		m_messageConsole->Append(wxString::Format("PCI_DIO48H board # = %d", m_PCI_DIO48H_Object.DIO_board_num));
		m_messageConsole->Append(wxString::Format("PCI_DIO48H base address (chip 1) = 0x%04X",
							     m_PCI_DIO48H_Object.DIO_base_address));
		m_messageConsole->Append(wxString::Format("RDX base address = 0x%04X", m_RDX_base_address));

		// Spit out any errors from setting up the digital ports.
		vector <wxString>::iterator iter;
		for (iter = errorStrings.begin(); iter != errorStrings.end(); iter++) {
			m_messageConsole->Append(*iter);
		}

		m_messageConsole->Append("--------------------------------------------------------------------------------");
	}	
}


void MoogDotsCom::OpenTempo()
{
	m_tempoHandle = dx_open_recv(m_RDX_base_address, m_tempoBuffer, sizeof(m_tempoBuffer));
	m_tempoErr = dx_reset(m_tempoHandle);
}


void MoogDotsCom::CloseTempo()
{
	dx_close(m_tempoHandle);
}


void MoogDotsCom::SetConsolePointer(wxListBox *messageConsole)
{
	m_messageConsole = messageConsole;
}


void MoogDotsCom::SetVerbosity(bool value)
{
	m_verboseMode = value;
}


StarField MoogDotsCom::createStarField()
{
	// Create a StarField structure that describes the GL starfield.
	StarField s;
	s.dimensions = g_pList.GetVectorData("STAR_VOLUME");
	s.density = g_pList.GetVectorData("STAR_DENSITY")[0];
	s.triangle_size = g_pList.GetVectorData("STAR_SIZE");
	//s.drawTarget = g_pList.GetVectorData("TARG_CROSS")[0];
	s.drawFixationPoint = g_pList.GetVectorData("FP_ON")[0];
	s.drawTarget1 = g_pList.GetVectorData("TARG1_ON")[0];
	s.drawTarget2 = g_pList.GetVectorData("TARG2_ON")[0];
	s.drawBackground = g_pList.GetVectorData("BACKGROUND_ON")[0];
	s.targetSize = g_pList.GetVectorData("TARGET_SIZE")[0];
	s.starLeftColor = g_pList.GetVectorData("STAR_LEYE_COLOR");
	s.starRightColor = g_pList.GetVectorData("STAR_REYE_COLOR");
	s.luminance = g_pList.GetVectorData("STAR_LUM_MULT")[0];
	s.lifetime = (int)g_pList.GetVectorData("STAR_LIFETIME")[0];
	s.probability = g_pList.GetVectorData("STAR_MOTION_COHERENCE")[0];
	s.use_lifetime = g_pList.GetVectorData("STAR_LIFETIME_ON")[0];
	s.useCutout = g_pList.GetVectorData("USE_CUTOUT").at(0) ? true : false;
	s.cutoutRadius = g_pList.GetVectorData("CUTOUT_RADIUS").at(0);
	s.drawMode = g_pList.GetVectorData("DRAW_MODE").at(0);

	// Fixation point.
	vector<double> a;
	a.push_back(g_pList.GetVectorData("TARG_XCTR")[0]);
	a.push_back(g_pList.GetVectorData("TARG_YCTR")[0]);
	a.push_back(g_pList.GetVectorData("TARG_ZCTR")[0]);
	s.fixationPointLocation = a;

	// Target 1.
	vector<double> b;
	b.push_back(g_pList.GetVectorData("TARG_XCTR")[1]);
	b.push_back(g_pList.GetVectorData("TARG_YCTR")[1]);
	b.push_back(g_pList.GetVectorData("TARG_ZCTR")[1]);
	s.targ1Location = b;

	// Target  2.
	vector<double> c;
	c.push_back(g_pList.GetVectorData("TARG_XCTR")[2]);
	c.push_back(g_pList.GetVectorData("TARG_YCTR")[2]);
	c.push_back(g_pList.GetVectorData("TARG_ZCTR")[2]);
	s.targ2Location = c;

	// Luminous Multi for fixation point, target 1 and target 2
	vector<double> d;
	d.push_back(g_pList.GetVectorData("TARG_LUM_MULT")[0]);
	d.push_back(g_pList.GetVectorData("TARG_LUM_MULT")[1]);
	d.push_back(g_pList.GetVectorData("TARG_LUM_MULT")[2]);
	s.targLumMult = d;

	s.targXsize = g_pList.GetVectorData("TARG_XSIZ");	// X-dimension of FP & targets // following add by Johnny - 11/6/08
	s.targYsize = g_pList.GetVectorData("TARG_YSIZ");	// Y-dimension of FP & targets
	s.targShape = g_pList.GetVectorData("TARG_SHAPE");	// shape of FP & targets: ELLIPSE or RECTANGLE
	s.targRlum = g_pList.GetVectorData("TARG_RLUM");	// red luminance of targets/FP: 0 -> 1
	s.targGlum = g_pList.GetVectorData("TARG_GLUM");	// green luminance of targets/FP: 0 -> 1
	s.targBlum = g_pList.GetVectorData("TARG_BLUM");	// blue luminance of targets/FP: 0 -> 1
	return s;
}


Frustum MoogDotsCom::createFrustum()
{
	vector<double> eyeOffsets = g_pList.GetVectorData("EYE_OFFSETS");
	vector<double> headCenter = g_pList.GetVectorData("HEAD_CENTER");

	// Create a new Frustum structure that describes the GL space we'll be working with.
	Frustum f;
	f.screenWidth = g_pList.GetVectorData("SCREEN_DIMS")[0];					// Screen width
	f.screenHeight = g_pList.GetVectorData("SCREEN_DIMS")[1];					// Screen height
	f.clipNear = g_pList.GetVectorData("CLIP_PLANES")[0];						// Near clipping plane.
	f.clipFar = g_pList.GetVectorData("CLIP_PLANES")[1];						// Far clipping plane.
	f.eyeSeparation = g_pList.GetVectorData("IO_DIST")[0];						// Distance between eyes.
	f.camera2screenDist = CENTER2SCREEN - eyeOffsets.at(1) - headCenter.at(1);	// Distance from monkey to screen.
	f.worldOffsetX = eyeOffsets.at(0) + headCenter.at(0);						// Horizontal world offset.
	f.worldOffsetZ = eyeOffsets.at(2) + headCenter.at(2);						// Vertical world offset.

#if JOHNNY_WORKSTATION
	f.screenWidth = 37.5;					// Screen width
	f.screenHeight = 30.0;					// Screen height
#endif

#if !DUAL_MONITORS
	f.screenWidth = 22.5;					// Screen width
	f.screenHeight = 24.5;					// Screen height
#endif

	return f;
}


void MoogDotsCom::UpdateGLScene(bool doSwapBuffers)
{
	bool drawTarget = true, drawBackground = true;
	GLPanel *g = m_glWindow->GetGLPanel();

	// Make sure that the glWindow actually has been created before we start
	// messing with it.
	if (m_glWindowExists == false) {
		return;
	}

	// Create a new Frustum and StarField based on the parameter list.
	Frustum f = createFrustum();
	StarField s = createStarField();

	// Grab pointers to the Frustum and StarField.
	Frustum *pf = g->GetFrustum();
	StarField *ps = g->GetStarField();

	// If the new Frustum and StarField are different from the old ones, then
	// we'll update them.
	if (compareFrustums(f, *pf) == false) {
		g->SetFrustum(f);
	}

	//if (compareStarFields(s, *ps) == false) {
		g->SetStarField(s);
	//}

	if (g_pList.GetVectorData("DRAW_MODE").at(0)==0.0) 
		g->SetStarField(s);

	// Target size, target on/off, and background on/off are not checked in the
	// compare functions so that we can manipulate them without forcing the
	// whole starfield to be redrawn.
	//ps->drawTarget = g_pList.GetVectorData("TARG_CROSS").at(0);
	ps->drawFixationPoint = g_pList.GetVectorData("FP_ON").at(0);
	ps->drawTarget1 = g_pList.GetVectorData("TARG1_ON").at(0);
	ps->drawTarget2 = g_pList.GetVectorData("TARG2_ON").at(0);
	ps->drawBackground = g_pList.GetVectorData("BACKGROUND_ON").at(0);
	ps->targetSize = g_pList.GetVectorData("TARGET_SIZE").at(0);
	ps->starLeftColor = g_pList.GetVectorData("STAR_LEYE_COLOR");
	ps->starRightColor = g_pList.GetVectorData("STAR_REYE_COLOR");
	ps->luminance = g_pList.GetVectorData("STAR_LUM_MULT").at(0);
	ps->lifetime = (int)g_pList.GetVectorData("STAR_LIFETIME").at(0);
	ps->probability = g_pList.GetVectorData("STAR_MOTION_COHERENCE").at(0);
	ps->use_lifetime = g_pList.GetVectorData("STAR_LIFETIME_ON").at(0);
	ps->useCutout = g_pList.GetVectorData("USE_CUTOUT").at(0) ? true : false;
	ps->cutoutRadius = g_pList.GetVectorData("CUTOUT_RADIUS").at(0);

	// Create a conversion factor to convert from TEMPO degrees
	// into centimeters for OpenGL.
	double deg2cm = tan(deg2rad(1.0))*(CENTER2SCREEN -
									   g_pList.GetVectorData("HEAD_CENTER").at(1) -
									   g_pList.GetVectorData("EYE_OFFSETS").at(1));

	// Fixation point.
	vector<double> a;
	a.push_back(g_pList.GetVectorData("TARG_XCTR").at(0)*deg2cm);
	a.push_back(g_pList.GetVectorData("TARG_YCTR").at(0)*deg2cm);
	a.push_back(g_pList.GetVectorData("TARG_ZCTR").at(0)*deg2cm);
	ps->fixationPointLocation = a;

	// Target 1.
	vector<double> b;
	b.push_back(g_pList.GetVectorData("TARG_XCTR").at(1)*deg2cm);
	b.push_back(g_pList.GetVectorData("TARG_YCTR").at(1)*deg2cm);
	b.push_back(g_pList.GetVectorData("TARG_ZCTR").at(1)*deg2cm);
	ps->targ1Location = b;

	// Target  2.
	vector<double> c;
	c.push_back(g_pList.GetVectorData("TARG_XCTR").at(2)*deg2cm);
	c.push_back(g_pList.GetVectorData("TARG_YCTR").at(2)*deg2cm);
	c.push_back(g_pList.GetVectorData("TARG_ZCTR").at(2)*deg2cm);
	ps->targ2Location = c;

	// Luminous Multi for fixation point, target 1 and target 2
	vector<double> d;
	d.push_back(g_pList.GetVectorData("TARG_LUM_MULT")[0]);
	d.push_back(g_pList.GetVectorData("TARG_LUM_MULT")[1]);
	d.push_back(g_pList.GetVectorData("TARG_LUM_MULT")[2]);
	ps->targLumMult = d;

	d.clear();
	for(int i=0; i<3; i++) d.push_back(g_pList.GetVectorData("TARG_XSIZ").at(i)*deg2cm);
	ps->targXsize = d;

	// Y-dimension of FP & targets
	d.clear();
	for(int i=0; i<3; i++) d.push_back(g_pList.GetVectorData("TARG_YSIZ").at(i)*deg2cm);
	ps->targYsize = d;

	// find vertex of ellipse shape and store in vector
	for(int i=0; i<3; i++)
	{
		int j=0;
		for(int k=0; k<DRAW_TARG_SLICES; k++)
		{
			ps->targVertex[i][j++] = ps->targXsize.at(i)*cos(k*2*PI/DRAW_TARG_SLICES)/2.0;
			ps->targVertex[i][j++] = ps->targYsize.at(i)*sin(k*2*PI/DRAW_TARG_SLICES)/2.0;
			ps->targVertex[i][j++] = 0;
		}
	}

	ps->targShape = g_pList.GetVectorData("TARG_SHAPE");// shape of FP & targets: ELLIPSE or RECTANGLE
	ps->targRlum = g_pList.GetVectorData("TARG_RLUM");	// red luminance of targets/FP: 0 -> 1
	ps->targGlum = g_pList.GetVectorData("TARG_GLUM");	// green luminance of targets/FP: 0 -> 1
	ps->targBlum = g_pList.GetVectorData("TARG_BLUM");	// blue luminance of targets/FP: 0 -> 1


	g->SetupParameters();

	// Output signal to CED
	stimAnalogOutput = g_pList.GetVectorData("STIM_ANALOG_OUTPUT").at(0);
	stimAnalogMult = g_pList.GetVectorData("STIM_ANALOG_MULT").at(0);

	// set visual syncronisation
	if (g_pList.GetVectorData("VISUAL_SYNC").at(0)){
		SetVSyncState(true);
		UseCustomTimer(true);
	}
	else {
		SetVSyncState(false);
		UseCustomTimer(false);
	}

	if (doSwapBuffers) {
		// Re-Render the scene.
		g->Render();

		// If we're in the main thread then call the GLPanel's SwapBuffers() function
		// because that function references the context made in the main thread.
		// Otherwise, we need to swap the buffers based on the context made in the
		// communications thread.
		if (wxThread::IsMain() == true) {
			g->SwapBuffers();
		}
		else {
			wglMakeCurrent((HDC)g->GetContext()->GetHDC(), m_threadGLContext);
			SwapBuffers((HDC)g->GetContext()->GetHDC());
		}
	}
}


bool MoogDotsCom::compareFrustums(Frustum a, Frustum b) const
{
	// Compare every element in the two Frustums.
	bool equalFrustums = a.camera2screenDist == b.camera2screenDist &&
						 a.clipNear == b.clipNear &&
						 a.clipFar == b.clipFar &&
						 a.eyeSeparation == b.eyeSeparation &&
						 a.screenHeight == b.screenHeight &&
						 a.screenWidth == b.screenWidth &&
						 a.worldOffsetX == b.worldOffsetX &&
						 a.worldOffsetZ == b.worldOffsetZ;

	return equalFrustums;
}

bool MoogDotsCom::compareStarFields(StarField a, StarField b) const
{
	bool equalStarFields;

	// Compare every element in the two StarFields.
	equalStarFields = a.density == b.density &&
					  a.dimensions == b.dimensions &&
					  a.triangle_size == b.triangle_size &&
					  a.drawMode == b.drawMode;

	return equalStarFields;
}


void MoogDotsCom::Sync()
{
	// Sync to a SwapBuffers() call.
	SetVSyncState(true);
	wglMakeCurrent((HDC)m_glWindow->GetGLPanel()->GetContext()->GetHDC(), m_threadGLContext);
	SwapBuffers((HDC)m_glWindow->GetGLPanel()->GetContext()->GetHDC());
	SetVSyncState(false);
	Delay(m_delay);
}


void MoogDotsCom::ThreadInit(void)
{
	// Setup the rendering context for the thread.  Every thread has to
	// have its own rendering context.
	if (m_glWindowExists) {
		m_threadGLContext = wglCreateContext((HDC)m_glWindow->GetGLPanel()->GetContext()->GetHDC());

		// Make sure that we got a valid handle.
		if (m_threadGLContext == NULL) {
			wxMessageDialog d(NULL, "ThreadInit: Couldn't create GL Context.", "GL Error");
			d.ShowModal();
		}

		if (wglMakeCurrent((HDC)m_glWindow->GetGLPanel()->GetContext()->GetHDC(), m_threadGLContext) == FALSE) {
			wxMessageDialog d(NULL, "ThreadInit: Couldn't MakeCurrent.", "GL ERROR");
			d.ShowModal();
		}

		// Initialize the GL Session.
		m_glWindow->GetGLPanel()->InitGL();
	}

	m_data.index = 0;
	m_recordOffset = 0;
	m_recordIndex = 0;
	m_grabIndex = 0;

	// If we're using pipes, wait for a connection.
	if (m_connectionType == MoogDotsCom::ConnectionType::Pipes) {
		ConnectNamedPipe(m_pipeHandle, &m_overlappedEvent);

		// Wait for the pipe to get signaled.
		m_messageConsole->InsertItems(1, &wxString("Waiting for client connection..."), 0);

#if MINI_MOOG_SYSTEM
		wxBusyInfo wait("Waiting for client connection...");
#endif	

		WaitForSingleObject(m_overlappedEvent.hEvent, INFINITE);
		m_messageConsole->InsertItems(1, &wxString("Connected!"), 0);

		// Check the result.
		DWORD junk;
		if (GetOverlappedResult(m_pipeHandle, &m_overlappedEvent, &junk, FALSE) == 0) {
			wxMessageDialog d(NULL, "GetOverlappedResult failed.");
			d.ShowModal();
		}

		ResetEvent(m_overlappedEvent.hEvent);
	}

#if MINI_MOOG_SYSTEM
	else{ // In Mini Moog, Tempo also need a little time to connect
		wxBusyInfo wait("Waiting for client connection...");
	}
#endif

#if USE_MATLAB_RDX
	// Create the Matlab RDX communication object if it doesn't exist.
	if (m_matlabRDX == NULL) {
		m_matlabRDX = new CMatlabRDX(m_PCI_DIO48H_Object.DIO_board_num);
	}
	m_matlabRDX->InitClient();
#endif
}


#if USE_MATLAB

void MoogDotsCom::StartMatlab()
{
	m_engine = engOpen(NULL);
}


void MoogDotsCom::CloseMatlab()
{
	engClose(m_engine);
}

void MoogDotsCom::StuffMatlab()
{
	stuffDoubleVector(m_sendStamp, "sendTimes");
	stuffDoubleVector(m_recordedHeave, "rHeave");
	stuffDoubleVector(m_recordedLateral, "rLateral");
	stuffDoubleVector(m_recordedSurge, "rSurge");
	stuffDoubleVector(m_recordedYaw, "rYaw");
	stuffDoubleVector(m_recordedPitch, "rPitch");
	stuffDoubleVector(m_recordedRoll, "rRoll");
	stuffDoubleVector(m_receiveStamp, "receiveTimes");

	// Stuff the command data into Matlab.
	stuffDoubleVector(m_data.X, "dataX");
	stuffDoubleVector(m_data.Y, "dataY");
	stuffDoubleVector(m_data.Z, "dataZ");

	// Stuff the interpolated, predicted data into Matlab.
	stuffDoubleVector(m_interpHeave, "iHeave");
	stuffDoubleVector(m_interpSurge, "iSurge");
	stuffDoubleVector(m_interpLateral, "iLateral");
	stuffDoubleVector(m_interpRotation, "iRotation");

	// Stuff the noise data.
	stuffDoubleVector(m_noise.X, "noiseX");
	stuffDoubleVector(m_noise.Y, "noiseY");
	stuffDoubleVector(m_noise.Z, "noiseZ");
	stuffDoubleVector(m_filteredNoise.X, "fnoiseX");
	stuffDoubleVector(m_filteredNoise.Y, "fnoiseY");
	stuffDoubleVector(m_filteredNoise.Z, "fnoiseZ");

	// Stuff rotation data.
	stuffDoubleVector(m_rotData.X, "yaw");
	stuffDoubleVector(m_rotData.Y, "pitch");
	stuffDoubleVector(m_rotData.Z, "roll");

	stuffDoubleVector(m_fpRotData.X, "fpA");
	stuffDoubleVector(m_fpRotData.Y, "fpE");

	stuffDoubleVector(m_recordedYaw, "ayaw");
	stuffDoubleVector(m_recordedYawVel, "ayawv");

	// Stuff CED Output
	stuffDoubleVector(m_gaussianTrajectoryData, "CEDInputPos");
	stuffDoubleVector(m_interpStimulusData, "CEDTransformPos");
    stuffDoubleVector(m_StimulusData, "CEDOutputVeloc");
#if SWAP_TIMER
	stuffDoubleVector(m_swapStamp, "swapTimes");
#endif

	if (g_pList.GetVectorData("DRAW_MODE").at(0) == 3.0){ // Dots' hollow sphere
		stuffDoubleVector(m_glWindow->GetGLPanel()->dotSizeDistribute, "dotSizeDistribute");
	}
}


void MoogDotsCom::stuffDoubleVector(vector<double> data, const char *variable)
{
	int i;
	mxArray *matData;

	// Create an mxArray large enough to hold the contents of the data vector.
	matData = mxCreateDoubleMatrix(1, data.size(), mxREAL);

	// Clear the variable if it exists.
	string s = "clear "; s += variable;
	engEvalString(m_engine, s.c_str());

	// Stuff the mxArray with the vector data.
	for (i = 0; i < (int)data.size(); i++) {
		mxGetPr(matData)[i] = data[i];
	}

	engPutVariable(m_engine, variable, matData);

	mxDestroyArray(matData);
}

void MoogDotsCom::stuffDoubleVector(vector<int> data, const char *variable)
{
	int i;
	mxArray *matData;

	// Create an mxArray large enough to hold the contents of the data vector.
	matData = mxCreateDoubleMatrix(1, data.size(), mxREAL);

	// Clear the variable if it exists.
	string s = "clear "; s += variable;
	engEvalString(m_engine, s.c_str());

	// Stuff the mxArray with the vector data.
	for (i = 0; i < (int)data.size(); i++) {
		mxGetPr(matData)[i] = data[i];
	}

	engPutVariable(m_engine, variable, matData);

	mxDestroyArray(matData);
}

#endif // USE_MATLAB


string MoogDotsCom::checkTempo()
{
	bool command_complete = false;
	char c;
	short nCnt, nErr;
	string command = "";		// Holds the string returned from the buffer.

	// If nothing is in the queue, don't do anything else.
	nCnt = dx_recv(m_tempoHandle);
	if (!nCnt) {
		return command;
	}

	// Until we get an complete command from the buffer, just loop.
	while (command_complete == false)
	{
		while (nCnt)
		{
			// Check for more characters.
			nErr = dx_getchar(m_tempoHandle, &c);
			if (!nErr) {
				break;			//no more chraracters
			}

			// Look to see if we've gotten to the end of the command.
			if ((c == '\n') || (c == '\r'))
			{
				command_complete = 1;
				break;
			}
			else {
				command += c;	// Add the character to the command string.
			}
		}

		if (command_complete == false) {
			nCnt = dx_recv(m_tempoHandle);
		}
	}

	return command;
}


bool MoogDotsCom::CheckForEStop()
{
	unsigned short digIn;			// Stores the read in estop bit.
	bool eStopActivated = false;	// Indicates if the estop sequence was activated.

	// Read the digital bit.
	cbDIn(ESTOP_IN_BOARDNUM, FIRSTPORTA, &digIn);

	// If it's currently high, but previously low, then a stop command has been issued.
	if ((digIn & 2) && m_previousBitLow == true) {
		eStopActivated = true;

		vector<double> stopVal, zeroCode;
		stopVal.push_back(2.0);
		zeroCode.push_back(0.0);
		
		m_previousBitLow = false;

		if (m_verboseMode) {
			m_messageConsole->InsertItems(1, &wxString("***** Fixation Break!"), 0);
		}

		// Turn off a bunch of visual stuff along with the movement stop.
		g_pList.SetVectorData("DO_MOVEMENT", stopVal);
		g_pList.SetVectorData("BACKGROUND_ON", zeroCode);
		g_pList.SetVectorData("FP_ON", zeroCode);
		g_pList.SetVectorData("TARG1_ON", zeroCode);
		g_pList.SetVectorData("TARG2_ON", zeroCode);

//#if CUSTOM_TIMER - Johnny 6/17/2007
		// Turn off sync pulse.
		m_doSyncPulse = false;
//#endif

#if USE_ANALOG_OUT_BOARD
		// Zero the analog out board.
		cbAOut(m_PCI_DIO48H_Object.DIO_board_num, 0, BIP10VOLTS, DASCALE(0.0));
		cbAOut(m_PCI_DIO48H_Object.DIO_board_num, 1, BIP10VOLTS, DASCALE(0.0));
#endif
	}

	// Reset the previous bit setting to low if the current bit state
	// is low.
	if ((digIn & 2) == 0) {
		m_previousBitLow = true;
	}

	return eStopActivated;
}


void MoogDotsCom::Control()
{
	string command = "";
	LARGE_INTEGER st, fi;
	double start, finish;
	bool stuffChanged = false;
	BOOL result;

	if(m_grabIndex > 0 && m_grabIndex < 121) 
		controlCounter++;
	else controlCounter=0;

	// Don't do anything if listen mode isn't enabled or the connection type is set to none.
	if (m_listenMode == false || m_connectionType == MoogDotsCom::ConnectionType::None) {
		return;
	}

#if ESTOP
	// Check to see if the estop bit has been set.  If it has, then CheckForEStop will
	// take care of turning the display off and setting the trajectory to be a buffered stop.
	stuffChanged = CheckForEStop();
#endif

	if (m_connectionType == MoogDotsCom::ConnectionType::Pipes) {
		// Request to read from the pipe if we already haven't done so.
		if (m_iocomplete == true) {
			result = ReadFile(m_pipeHandle, m_pipeBuff, (DWORD)PIPE_BUFSIZE, NULL, &m_overlappedEvent);
			m_iocomplete = false;

			// Check to see if there was a critical error.
			if (result == 0 && GetLastError() != ERROR_IO_PENDING) {
				wxString e = wxString::Format("*** ReadFile failed with error: %d", GetLastError());
				m_messageConsole->InsertItems(1, &e, 0);
			}
		}
	}

	QueryPerformanceCounter(&st);
	start = static_cast<double>(st.QuadPart) / static_cast<double>(m_freq.QuadPart) * 1000.0;

	try {
		// Loop for a maximum of CONTROL_LOOP_TIME to get as much stuff from the Tempo buffer as possible.
		do {
			// Do Tempo stuff if we have a valid tempo handle and we actually received something on the buffer.
#if USE_MATLAB_RDX
			if (m_matlabRDX->ReadString(1.0, 64, &command) > 0) {
#else
			// Depending on the connection type, we check for the presence of new data
			// in different ways.  
			bool commandReceived = false;
			if (m_connectionType == MoogDotsCom::ConnectionType::Pipes) {
				// Check to see if any new data has arrived.
				CheckPipes();
				commandReceived = !m_commandQueue.empty();

				// If the command queue isn't empty, read the command off the front of
				// the queue and pop it off.
				if (commandReceived) {
					command = m_commandQueue.front();
					m_commandQueue.pop();
				}
			}
			else if (m_connectionType == MoogDotsCom::ConnectionType::Tempo) {
				commandReceived = m_tempoHandle >= 0 && (command = checkTempo()) != "";
			}

			if (commandReceived) {
#endif
				stuffChanged = true;

				// Parses the command received from Tempo or the pipes.
				ProcessCommand(command);
			}
			else {
				start = 0.0;
			}

			QueryPerformanceCounter(&fi);
			finish = static_cast<double>(fi.QuadPart) / static_cast<double>(m_freq.QuadPart) * 1000.0;
		} while ((finish - start) < CONTROL_LOOP_TIME);
	}
	catch (exception &e) {
		stuffChanged = false;
		m_messageConsole->InsertItems(1, &wxString("*** Serious screwup detected!"), 0);
	}

	// Only waste time updating stuff if we actually received valid data from Tempo.
	if (stuffChanged) {
		// Updates the GL scene.
		UpdateGLScene(true);

		// Updates basic Moog trajectory movement, including whether or not to move.
		UpdateMovement();
	}
} // End void MoogDotsCom::Control()


void MoogDotsCom::CheckPipes()
{
	BOOL result;
	DWORD numBytesRead;
	string command;
	int i = 0, offset = 0;
	bool keepReading = true;

	// Check to see if ReadFile() finished.
	result = WaitForSingleObject(m_overlappedEvent.hEvent, 0);
	switch (result)
	{
		// ReadFile() hasn't finished yet.
	case WAIT_TIMEOUT:
		break;

		// Object is signaled, i.e. ReadFile() completed.
	case WAIT_OBJECT_0:
		// Get the result of the ReadFile() call and reset the signal.
		if (GetOverlappedResult(m_pipeHandle, &m_overlappedEvent, &numBytesRead, FALSE) == 0) {
			wxString es = wxString::Format("GetOverlappedResult failed with error: %d", GetLastError());
			m_messageConsole->InsertItems(1, &es, 0);

			//Automatically disconnect to the pipes when pipes disconnect from Spike2
			wxCommandEvent event;
			m_mainFrame->OnMenuToolsDisconnectPipes(event);
		}
		ResetEvent(m_overlappedEvent.hEvent);
		m_iocomplete = true;

		// Find the newline character and copy over the buffer into the string
		// before that character.
		//while (keepReading) {
		//	if (m_pipeBuff[i] == '\n' || m_pipeBuff[i] == '\r') {
		//		keepReading = false;
		//		i--;
		//	}
		//	i++;
		//}
		//command.assign(m_pipeBuff, i);

		for (i = 0; i < static_cast<int>(numBytesRead); i++) {
			if (m_pipeBuff[i] == '\n' || m_pipeBuff[i] == '\r') {
				command.assign(m_pipeBuff, offset, i-offset);

				// Spike2 uses a '\r\n' to represent a newline, so move the iterator
				// past the 2nd element of the newline.
				i++;

				// The offset of the next command string will be one past the current
				// iterator value.
				offset = i + 1;

				m_commandQueue.push(command);
			}
		}

		break;

	case WAIT_ABANDONED:
		m_messageConsole->InsertItems(1, &wxString("Wait abandoned."), 0);
		break;
	} // End switch (result)
}


void MoogDotsCom::ProcessCommand(string command)
{
	string keyword, param;
	int spaceIndex, tmpIndex, tmpEnd;
	double convertedValue;
	vector<double> commandParams;

	// This removes lines in the message console if it is getting too long.
	int numItems = m_messageConsole->GetCount();
	if (numItems >= MAX_CONSOLE_LENGTH) {
		for (int i = 0; i <= numItems-MAX_CONSOLE_LENGTH; i++) {
			m_messageConsole->Delete(m_messageConsole->GetCount()-1);
		}
	}

	// Put the command in the message console.
	if (m_verboseMode) {
		wxString s(command.c_str());
		m_messageConsole->InsertItems(1, &s, 0);
	}

	const char *ass = command.c_str();

	// Grab the keyword from the command.
	spaceIndex = command.find(" ", 0);

	// If we don't get a valid index, then we've likely gotten a command
	// for another program.  In that case, we don't parse the command string
	// and just set the keyword to "invalid" so that we ignore whatever
	// we just received.
	if (spaceIndex != string::npos) {
		keyword = command.substr(0, spaceIndex);

		// Loop and grab the parameters from the command string.
		do {
			tmpIndex = command.find_first_not_of(" ", spaceIndex+1);
			tmpEnd = command.find(" ", tmpIndex);

			// If someone accidentally put a space at the end of the
			// command string, then we want to skip extracting out a
			// parameter value.
			if (tmpIndex != string::npos) {
				if (tmpEnd != string::npos) {
					spaceIndex = tmpEnd;

					// Pull out the substring with the number in it.
					param = command.substr(tmpIndex, tmpEnd - tmpIndex);
				}
				else {
					// Pull out the substring with the number in it.
					param = command.substr(tmpIndex, command.size()-1);
				}

				// Convert the string to a double and stuff it in the vector.
				convertedValue = atof(param.c_str());
				commandParams.push_back(convertedValue);
			}
		} while (tmpEnd != string::npos);
	}
	else {
		keyword = "invalid";
	}

	// Set the parameter data if it's supposed to be in the parameter list.
	if (g_pList.Exists(keyword) == true) {
		if (g_pList.IsVariable(keyword) == false) {
			// Make sure that we don't have a parameter count mismatch.  This could
			// cause unexpected results.
			if (static_cast<int>(commandParams.size()) != g_pList.GetParamSize(keyword)) {
				if (m_verboseMode) {
					wxString s = wxString::Format("INVALID PARAMETER VALUES FOR %s.", keyword);
					m_messageConsole->InsertItems(1, &s, 0);
				}
			}
			else {
				g_pList.SetVectorData(keyword, commandParams);

//#if CUSTOM_TIMER - Johnny 6/17/07
				// If we get the BACKGROUND_ON signal, toggle the sync pulses on or off.
				//if (keyword.compare("BACKGROUND_ON") == 0) {
				if (m_customTimer && keyword.compare("BACKGROUND_ON") == 0) {
					if (commandParams.at(0) == 1.0) {
						m_doSyncPulse = true;
					}
					else {
						m_doSyncPulse = false;
					}
				}else if(keyword.compare("ADJUSTED_OFFSET") == 0){
					m_glWindow->GetGLPanel()->adjusted_offset=g_pList.GetVectorData("ADJUSTED_OFFSET").at(0);
					m_glWindow->GetGLPanel()->adjusted_ele_offset=g_pList.GetVectorData("ADJUSTED_OFFSET").at(1);
					UpdateGLScene(true);
				}
//#endif
			} // End if (g_pList.Exists(keyword) == true)
		}
		else {
			g_pList.SetVectorData(keyword, commandParams);
		} // End if (g_pList.IsVariable(keyword) == false)
	}
	else {	// Didn't find keyword in the parameter list.
		wxString s = wxString::Format("UNKNOWN COMMAND: %s.", command.c_str());
		m_messageConsole->InsertItems(1, &s, 0);
	} // End if (superFreak.empty() == false)
}


void MoogDotsCom::UpdateMovement()
{
	vector<double> zeroVector;
	int switchCode = static_cast<int>(g_pList.GetVectorData("DO_MOVEMENT").at(0));
	int i;

	zeroVector.push_back(0.0);

	if (g_pList.GetVectorData("GO_TO_ZERO").at(0) == 0.0) {
		DATA_FRAME startFrame;

		FILE *fp;

		// Grab the movement parameters.
		vector<double> startPoint = g_pList.GetVectorData("M_ORIGIN");
		vector<double> rotStartPoint = g_pList.GetVectorData("ROT_ORIGIN");
		startFrame.lateral = startPoint.at(0); startFrame.surge = startPoint.at(1); startFrame.heave = startPoint.at(2);
		startFrame.yaw = rotStartPoint.at(0); startFrame.pitch = rotStartPoint.at(1); startFrame.roll = rotStartPoint.at(2);
		double elevation = g_pList.GetVectorData("M_ELEVATION").at(0);
		double azimuth = g_pList.GetVectorData("M_AZIMUTH").at(0);
//#if USING_FISHEYE
//		if(azimuth!=90.0)
//			elevation +=m_moogCom->m_glWindow->GetGLPanel()->adjusted_ele_offset;
//#endif
		double magnitude = g_pList.GetVectorData("M_DIST").at(0);
		double duration = g_pList.GetVectorData("M_TIME").at(0);
		double sigma = g_pList.GetVectorData("M_SIGMA").at(0);
		motionType = g_pList.GetVectorData("MOTION_TYPE").at(0);

		switch (switchCode)
		{
		case 0:	// Do nothing
			break;

		case 1:	// Start
			// Choose what kind of motion we'll be doing.
			switch (motionType)
			{
			case 0:
				// Calculate the trajectory to move the motion base into start position and along the vector.
				CalculateGaussianMovement(&startFrame, elevation, azimuth, magnitude, duration, sigma, true);

				// Generate the analog output for the velocity if needed.
				if (g_pList.GetVectorData("OUTPUT_VELOCITY").at(0)) {
					CalculateStimulusOutput();
				}
				break;
			case 1:
				CalculateRotationMovement();
				break;
			case 2:
				CalculateSinusoidMovement();
				break;
			case 3:
				Calculate2IntervalMovement();
				break;
			case 4:
				CalculateTiltTranslationMovement();
				break;
			case 5:
				CalculateGaborMovement();
				break;
			case 6:
				CalculateStepVelocityMovement();
				break;
			}

			fp=fopen("C:\\Data\\mpos","at");
			fprintf(fp,"DO_MOVEMENT\n");// %f\n", getCurrentTimeInSec());
			fclose(fp);


			// This keeps the program from calculating the Gaussian movement over and over again.
			g_pList.SetVectorData("DO_MOVEMENT", zeroVector);

			// Output signal to CED
			stimAnalogOutput = g_pList.GetVectorData("STIM_ANALOG_OUTPUT").at(0);
			stimAnalogMult = g_pList.GetVectorData("STIM_ANALOG_MULT").at(0);
			m_glWindow->GetGLPanel()->redrawCallList = true;

#if MINI_MOOG_SYSTEM
			SetFuncExecute(RECEIVE_FUNC | COMPUTE_FUNC);
#else
			ThreadDoCompute(RECEIVE_COMPUTE | COMPUTE);
#endif
			break;

		case 2:	// Stop
			vector<double> x;
			double  ixv = 0.0, iyv = 0.0, izv = 0.0,		// Instananeous velocities (m/frame)
					iyawv = 0.0, ipitchv = 0.0, irollv = 0.0;

			// This will keep the Compute() function from trying to draw predicted data,
			// and use real feedback instead.
			m_recordOffset = SPEED_BUFFER_SIZE;

			// Get the current position of each axis.
			DATA_FRAME currentFrame;
			THREAD_GET_DATA_FRAME(&currentFrame);

			// Calculate the instantaneous velocity for each axis.
			ixv = currentFrame.lateral - m_previousPosition.lateral;
			iyv = currentFrame.surge - m_previousPosition.surge;
			izv = currentFrame.heave - m_previousPosition.heave;
			iyawv = currentFrame.yaw - m_previousPosition.yaw;
			ipitchv = currentFrame.pitch - m_previousPosition.pitch;
			irollv = currentFrame.roll - m_previousPosition.roll;

			// Reset the movement data.
			nmClearMovementData(&m_data);
			nmClearMovementData(&m_rotData);

			// Create buffered movement data.
			for (i = 0; i < SPEED_BUFFER_SIZE; i++) {
				// Translational movement data.
				currentFrame.lateral += ixv * m_speedBuffer[i];
				currentFrame.surge += iyv * m_speedBuffer[i];
				currentFrame.heave += izv * m_speedBuffer[i];
				m_data.X.push_back(currentFrame.lateral);
				m_data.Y.push_back(currentFrame.surge);
				m_data.Z.push_back(currentFrame.heave);

				// Rotational movement data.
				currentFrame.yaw += iyawv * m_speedBuffer[i];
				currentFrame.pitch += ipitchv * m_speedBuffer[i];
				currentFrame.roll += irollv * m_speedBuffer[i];
				m_rotData.X.push_back(currentFrame.yaw);
				m_rotData.Y.push_back(currentFrame.pitch);
				m_rotData.Z.push_back(currentFrame.roll);
			}

			// This keeps the program from calculating the stop movement over and over again.
			g_pList.SetVectorData("DO_MOVEMENT", zeroVector);

			break;
		};
	} // End if (g_pList.GetVectorData("GO_TO_ZERO")[0] == 0.0)
	else {
		vector<double> startPoint = g_pList.GetVectorData("M_ORIGIN");
		DATA_FRAME startFrame;
		startFrame.lateral = startPoint.at(0);
		startFrame.surge = startPoint.at(1);
		startFrame.heave = startPoint.at(2);
		startFrame.yaw = 0.0f;
		startFrame.pitch = 0.0f;
		startFrame.roll = 0.0f;

		// This will move the motion base from its current position to (0, 0, 0).
		CalculateGaussianMovement(&startFrame, 0.0, 0.0, 0.0, 0.0, 0.0, false);

		g_pList.SetVectorData("GO_TO_ZERO", zeroVector);
		g_pList.SetVectorData("DO_MOVEMENT", zeroVector);

#if USE_ANALOG_OUT_BOARD
		// Zero out PCI-DDA02/16 board.
		cbAOut(m_PCI_DIO48H_Object.DIO_board_num, 0, BIP10VOLTS, DASCALE(0.0));
		cbAOut(m_PCI_DIO48H_Object.DIO_board_num, 1, BIP10VOLTS, DASCALE(0.0));
#endif

#if MINI_MOOG_SYSTEM
		SetFuncExecute(RECEIVE_FUNC | COMPUTE_FUNC);
#else
		ThreadDoCompute(RECEIVE_COMPUTE | COMPUTE);
#endif
	}
}


void MoogDotsCom::Compute()
{
	if (m_data.index < static_cast<int>(m_data.X.size())) {
#if MINI_MOOG_SYSTEM
		GetDataFrame(&m_previousPosition);
#else
		ThreadGetAxesPositions(&m_previousPosition);
#endif

		// Setup the frame to send to the Moog.
		DATA_FRAME moogFrame;
		moogFrame.lateral = static_cast<float>(m_data.X.at(m_data.index));
		moogFrame.surge = static_cast<float>(m_data.Y.at(m_data.index));
		moogFrame.heave = static_cast<float>(m_data.Z.at(m_data.index));
		moogFrame.yaw = static_cast<float>(m_rotData.X.at(m_data.index));
		moogFrame.pitch = static_cast<float>(m_rotData.Y.at(m_data.index));
		moogFrame.roll = static_cast<float>(m_rotData.Z.at(m_data.index));
		SET_DATA_FRAME(&moogFrame);

		// Increment the counter which we use to pull data.
		m_data.index++;

		// Record the last send time's stamp.
#if USE_MATLAB
#if MINI_MOOG_SYSTEM
		m_sendStamp.push_back(GetSendTime());
#else
		m_sendStamp.push_back(ThreadGetSendTime());
#endif
#endif

#if !RECORD_MODE
		// Grab the shifted and interpolated data to draw.
		if (m_data.index > m_recordOffset && m_grabIndex < static_cast<int>(m_interpLateral.size())) {
//#if !CUSTOM_TIMER - Johnny 6/17/07
			if(!m_customTimer){
				// Send out a sync pulse only after the 1st frame of the trajectory has been
				// processed by the platform.  This equates to the 2nd time into this section
				// of the function.
#if FIRST_PULSE_ONLY
				if (m_data.index == m_recordOffset + 1) {
#else
				if (m_data.index > m_recordOffset + 1) {
#endif // FIRST_PULSE_ONLY
					cbDOut(PULSE_OUT_BOARDNUM, FIRSTPORTB, 1);
					cbDOut(PULSE_OUT_BOARDNUM, FIRSTPORTB, 0);
				}
			}
//#endif
			if (m_glWindowExists) {
				double azimuth = 0.0, elevation = 0.0;

				// Grab a pointer to the GLPanel.
				GLPanel *glPanel = m_glWindow->GetGLPanel();
#if DOF_MODE
				//  Set the translation components to the camera.
				int size = m_interpLateral.size();
				// for debug only
				//double a = m_interpLateral[m_grabIndex], b = m_interpHeave[m_grabIndex], c = m_interpSurge[m_grabIndex];

				glPanel->SetLateral(m_interpLateral.at(m_grabIndex)*100.0);
				glPanel->SetHeave(m_interpHeave.at(m_grabIndex)*100.0 + GL_OFFSET);
				glPanel->SetSurge(m_interpSurge.at(m_grabIndex)*100.0);

				// If we're doing rotation, set the rotation data.
				if (m_setRotation == true) {
					//Johnny comment out the val, because it never use. And
					//when the index of m_glRotData (m_grabIndex) increase, it will out of boundary and program crash.
					//double val = m_glRotData.at(m_grabIndex);
					int tmp =  m_interpRotation.size();
					double angle = m_interpRotation.at(m_grabIndex);
					glPanel->SetRotationAngle(m_interpRotation.at(m_grabIndex));

					// Make sure we set the azimuth and elevation of the fixation point
					// if need to move the fixation window on the Tempo side.
					if (g_pList.GetVectorData("FP_ROTATE").at(0) >= 1.0) {
						azimuth = m_fpRotData.X.at(m_grabIndex);
						elevation = m_fpRotData.Y.at(m_grabIndex);
					}
				}

#if USE_ANALOG_OUT_BOARD
				// If we're not outputtinig the velocity profile, then we need to output
				// the horizontal and vertical position for Tempo to correctly draw the
				// fixation target.
				if(stimAnalogOutput==0){
					if (g_pList.GetVectorData("OUTPUT_VELOCITY").at(0) == 0.0) {
						unsigned short e = DASCALE(elevation),
									a = DASCALE(azimuth);
						cbAOut(m_PCI_DIO48H_Object.DIO_board_num, 0, BIP10VOLTS, DASCALE(azimuth));
						cbAOut(m_PCI_DIO48H_Object.DIO_board_num, 1, BIP10VOLTS, DASCALE(elevation));
					}
					else {
						OutputStimulusCurve(m_grabIndex);				
					}
				}

#endif
#else
				// This section is buggy, will probably have to rework it sometime.
				glPanel->SetLateral((m_interpLateral[m_grabIndex] + m_data.X[m_recordOffset])*100.0);
				glPanel->SetHeave((m_interpHeave[m_grabIndex] + m_data.Z[m_recordOffset])*100.0 + GL_OFFSET);
				glPanel->SetSurge((m_interpSurge[m_grabIndex] + m_data.Y[m_recordOffset])*100.0);
#endif // #if DOF_MODE
				// Render the scene.
				glPanel->Render();

//#if !CUSTOM_TIMER - Johnny 6/17/06
				if(!m_customTimer){
					// Swap the buffers.
					wglMakeCurrent((HDC)glPanel->GetContext()->GetHDC(), m_threadGLContext);
					SwapBuffers((HDC)glPanel->GetContext()->GetHDC());
				}
//#endif
				m_drawRegularFeedback = false;

				// If we only show visual display and Moog base doesn't move,
				// we need send some data back after we did SwapBuffers(),?
				// because it take 16.6ms to swap buffer of screen, but sending signal to Spike2 is quick.
				// rotation sine wave: rotation angle
				int channel = 0;
				double moogSignal = 0.0; // signal for Moog base
				double interpSignal = 0.0; // signal for drawing on screen
				double accelerSignal = 0.0; // Simulate accelerometer signal
				if(stimAnalogOutput!=0){
					switch( stimAnalogOutput ) 
					{
						case 1:
							if(m_grabIndex<(int)m_interpLateral.size()){
								moogSignal = moogFrame.lateral*100;
								interpSignal =  m_interpLateral.at(m_grabIndex)*100.0;
								accelerSignal =  m_accelerLateral.at(m_grabIndex)*100.0;
							}
							break;
						case 2:
							if(m_grabIndex<(int)m_interpHeave.size()){
								moogSignal = moogFrame.heave*100 + GL_OFFSET; // shift it back to zero.
								interpSignal =  m_interpHeave.at(m_grabIndex)*100.0 + GL_OFFSET;
								accelerSignal =  m_accelerHeave.at(m_grabIndex)*100.0 + GL_OFFSET;
							}
							break;
						case 3:
							if(m_grabIndex<(int)m_interpSurge.size()){
								moogSignal = moogFrame.surge*100;
								interpSignal =  m_interpSurge.at(m_grabIndex)*100.0;
								accelerSignal =  m_accelerSurge.at(m_grabIndex)*100.0;
							}
							break;
						case 4:
							if(m_grabIndex<(int)m_interpRotation.size()){
								moogSignal = moogFrame.yaw;
								interpSignal =  m_interpRotation.at(m_grabIndex);
								accelerSignal =  m_accelerRotation.at(m_grabIndex);
							}
							break;
						case 5:
							if(m_grabIndex<(int)m_interpRotation.size()){
								moogSignal = moogFrame.pitch;
								interpSignal =  m_interpRotation.at(m_grabIndex);
								accelerSignal =  m_accelerRotation.at(m_grabIndex);
							}
							break;
						case 6:
							if(m_grabIndex<(int)m_interpRotation.size()){
								moogSignal = moogFrame.roll;
								interpSignal =  m_interpRotation.at(m_grabIndex);
								accelerSignal =  m_accelerRotation.at(m_grabIndex);
							}
							break;
					}
					if(!m_customTimer){
						if (motionType == 6) // Step velocity
							cbAOut(m_PCI_DIO48H_Object.DIO_board_num, channel, BIP10VOLTS, DASCALE(stimAnalogMult*angularSpeed));
						else
							cbAOut(m_PCI_DIO48H_Object.DIO_board_num, channel, BIP10VOLTS, DASCALE(stimAnalogMult*accelerSignal));
					}
					else{
						if (motionType == 6) // Step velocity
							openGLsignal = stimAnalogMult*angularSpeed;
						else openGLsignal = stimAnalogMult*accelerSignal;
					}
					//cbAOut(m_PCI_DIO48H_Object.DIO_board_num, channel, BIP10VOLTS, DASCALE(moogSignal));
					//cbAOut(m_PCI_DIO48H_Object.DIO_board_num, channel+2, BIP10VOLTS, DASCALE(stimAnalogMult*interpSignal));
				}
			} // if (m_glWindowExists)

			m_grabIndex++;
		} // if (m_data.index > m_recordOffset && m_grabIndex < (int)m_interpLateral.size())
		else {
			//m_drawRegularFeedback = true;
			//turn off the back ground immediately
			if (motionType == 6 && m_grabIndex > 0){ // Step velocity
				vector<double> tmp;
				tmp.push_back(0.0);
				g_pList.SetVectorData("BACKGROUND_ON", tmp);
				g_pList.SetVectorData("FP_ON", tmp);
				for(int i=0; i<5; i++) tmp.push_back(0.0);
				g_pList.SetVectorData("FP_CROSS", tmp);
				UpdateGLScene(true);
				//g_pList.SetVectorData("SIN_FREQUENCY", tmp);
				openGLsignal = 0;
				cbAOut(m_PCI_DIO48H_Object.DIO_board_num, 0, BIP10VOLTS, DASCALE(0.0));
			}
		} // else if (m_data.index > m_recordOffset && m_grabIndex < (int)m_interpLateral.size())
#endif // #if !RECORD_MODE
	}
	else {
		//turn off the back ground immediately
		if (motionType == 6 && m_grabIndex > 0){ // Step velocity
			vector<double> tmp;
			tmp.push_back(0.0);
			g_pList.SetVectorData("BACKGROUND_ON", tmp);
			g_pList.SetVectorData("FP_ON", tmp);
			for(int i=0; i<5; i++) tmp.push_back(0.0);
			g_pList.SetVectorData("FP_CROSS", tmp);
			UpdateGLScene(true);
			//g_pList.SetVectorData("SIN_FREQUENCY", tmp);
			openGLsignal = 0;
			cbAOut(m_PCI_DIO48H_Object.DIO_board_num, 0, BIP10VOLTS, DASCALE(0.0));
		}

		// Stop telling the motion base to move, but keep on calling the ReceiveCompute() function.
#if MINI_MOOG_SYSTEM
		SetFuncExecute(RECEIVE_FUNC);
#else
		ThreadDoCompute(RECEIVE_COMPUTE);
#endif

#if USE_ANALOG_OUT_BOARD
		// If we're doing the analog CED output, zero out the board when we've finished the movement.
		if (g_pList.GetVectorData("OUTPUT_VELOCITY").at(0)) {
			cbAOut(m_PCI_DIO48H_Object.DIO_board_num, 0, BIP10VOLTS, DASCALE(0));
			cbAOut(m_PCI_DIO48H_Object.DIO_board_num, 1, BIP10VOLTS, DASCALE(0));
		}
#endif

		// Go back to regular drawing mode.
		//m_drawRegularFeedback = true;

#if DEBUG_DEFAULTS
		m_messageConsole->Append(wxString::Format("Compute finished, index = %d", m_data.index));
#endif
	} // if (m_data.index < (int)m_data.X.size())
} // End void MoogDotsCom::Compute()


#if !MINI_MOOG_SYSTEM
void MoogDotsCom::ReceiveCompute()
{
	// Get the latest return frame.
	DATA_FRAME returnFrame;
	returnFrame.heave = ThreadGetReturnedHeave();
	returnFrame.lateral = ThreadGetReturnedLateral();
	returnFrame.surge = ThreadGetReturnedSurge();
	returnFrame.yaw = ThreadGetReturnedYaw();
	returnFrame.pitch = ThreadGetReturnedPitch();
	returnFrame.roll = ThreadGetReturnedRoll();

#if RECORD_MODE
	// If we're actively putting movement data into the command buffer, store the
	// return data.  That is, if we're supposed to.
	if (m_data.index > m_recordOffset+0 && m_recordIndex < static_cast<int>(m_data.X.size())-m_recordOffset+11) {
		// Record the receive time of the return packet.
		m_receiveStamp.push_back(ThreadGetReceiveTime());

		m_recordIndex++;

		// We have to subtract off the startpoint of the Gaussian so that the recorded
		// data will always start from 0.
		m_recordedLateral.push_back((returnFrame.lateral - m_data.X[m_recordOffset])*100.0);
		m_recordedHeave.push_back((returnFrame.heave - m_data.Z[m_recordOffset])*100.0 + HEAVE_OFFSET);
		m_recordedSurge.push_back((returnFrame.surge - m_data.Y[m_recordOffset])*100.0);
		m_recordedYaw.push_back(returnFrame.yaw);
		m_recordedPitch.push_back(returnFrame.pitch);
		m_recordedRoll.push_back(returnFrame.roll);
	}
#else // #if RECORD_MODE
	if (m_drawRegularFeedback) {
		// Set the camera position.
		if (m_glWindowExists) {
			m_glWindow->GetGLPanel()->SetLateral(returnFrame.lateral*100.0);
			m_glWindow->GetGLPanel()->SetSurge(returnFrame.surge*100.0);
			m_glWindow->GetGLPanel()->SetHeave(returnFrame.heave*100.0 + HEAVE_OFFSET);

			m_glWindow->GetGLPanel()->Render();

//#if !CUSTOM_TIMER - Johnny 6/17/07
			if(!m_customTimer){
				// Swap the buffers.
				wglMakeCurrent((HDC)m_glWindow->GetGLPanel()->GetContext()->GetHDC(), m_threadGLContext);
				SwapBuffers((HDC)m_glWindow->GetGLPanel()->GetContext()->GetHDC());
			}
//#endif
		}
	}
#endif // #if RECORD_MODE

#if USE_MATLAB && !RECORD_MODE
	m_receiveStamp.push_back(ThreadGetReceiveTime());
#endif
} // ReceiveCompute()
#endif


void MoogDotsCom::CustomTimer()
{
//#if CUSTOM_TIMER - Johnny 6/17/07
	if(!m_customTimer) return;

#if SWAP_TIMER
	LARGE_INTEGER t;

	// Time stamp the SwapBuffers() call.
	QueryPerformanceCounter(&t);
	m_swapStamp.push_back((double)t.QuadPart / (double)m_freq.QuadPart * 1000.0);
#endif

	// Swap the buffers.
	wglMakeCurrent((HDC)m_glWindow->GetGLPanel()->GetContext()->GetHDC(), m_threadGLContext);
    SwapBuffers((HDC)m_glWindow->GetGLPanel()->GetContext()->GetHDC());

	// Send out a sync pulse.
	if (m_doSyncPulse == true) {
		cbDOut(PULSE_OUT_BOARDNUM, FIRSTPORTB, 1);
		cbDOut(PULSE_OUT_BOARDNUM, FIRSTPORTB, 0);
	}
	
	cbAOut(m_PCI_DIO48H_Object.DIO_board_num, 0, BIP10VOLTS, DASCALE(openGLsignal));
//#endif
}


string MoogDotsCom::replaceInvalidChars(string s)
{
	int i;

	for (i = 0; i < static_cast<int>(s.length()); i++) {
		switch (s[i]) {
			case '-':
				s[i] = 'n';
				break;
			case '.':
				s[i] = 'd';
				break;
		}
	}

	return s;
}


void MoogDotsCom::CalculateCircleMovement()
{
#if USE_MATLAB
	// Values that are only really used when taking debug and feedback data through Matlab.
#if RECORD_MODE
	m_recordedLateral.clear(); m_recordedHeave.clear(); m_recordedSurge.clear();
	m_recordedYaw.clear(); m_recordedPitch.clear(); m_recordedRoll.clear();
#endif
	m_sendStamp.clear(); m_receiveStamp.clear();
#endif

	// Do no move these initializations.  Their location in the function is very important for
	// threading issues.
	m_grabIndex = 0;
	m_recordIndex = 0;

	m_continuousMode = false;

	// This tells Compute() not to set any rotation info and the GLPanel not to try
	// to do any rotation transformations in Render().
	m_setRotation = false;
	m_glWindow->GetGLPanel()->DoRotation(false);

	// Make sure we don't rotate the fixation point.
	m_glWindow->GetGLPanel()->RotationType(0);

	// Moves the platform to start position.
	DATA_FRAME startFrame;
	startFrame.lateral = -0.1f;
	startFrame.surge = startFrame.yaw = startFrame.pitch = startFrame.roll = 0.0f;
	startFrame.heave = 0.0f;
	MovePlatform(&startFrame);

	m_recordOffset = m_data.X.size();

	nm3DDatum p, r;
	p.x = -0.1; p.y = 0.0; p.z = MOTION_BASE_CENTER;
	r.x = 0.0; r.y = 0.0; r.z = MOTION_BASE_CENTER;
	nmRotatePointAboutPoint(p, r, 0.0, 360.0*40.0, 120.0/60.0, 90.0, 0.0, &m_data, false);

	m_glData.X.clear(); m_glData.Y.clear(); m_glData.Z.clear();

	// Make sure the yaw, pitch, and roll components are filled with zeros for the
	// 2nd part of the movement.  Also, copy over the movement data to the glData structure.
	for (int i = m_recordOffset; i < (int)m_data.X.size(); i++) {
		m_rotData.X.push_back(0.0);
		m_rotData.Y.push_back(0.0);
		m_rotData.Z.push_back(0.0);
		m_glData.X.push_back(m_data.X.at(i));
		m_glData.Y.push_back(m_data.Y.at(i));
		m_glData.Z.push_back(m_data.Z.at(i));
	}

	GeneratePredictedData();
}


void MoogDotsCom::CalculateSinusoidMovement()
{
	// This tells Compute() not to set any rotation info and the GLPanel not to try
	// to do any rotation transformations in Render().
	m_setRotation = false;
	m_glWindow->GetGLPanel()->DoRotation(false);

	// Make sure we don't rotate the fixation point.
	m_glWindow->GetGLPanel()->RotationType(0);

	// Do no move these initializations.  Their location in the function is very important for
	// threading issues.
	m_grabIndex = 0;
	m_recordIndex = 0;

	// Clear the OpenGL data.
	nmClearMovementData(&m_glData);

	double duration,
		   freq = g_pList.GetVectorData("SIN_FREQUENCY").at(0);

	if (g_pList.GetVectorData("SIN_CONTINUOUS").at(0) == 1.0) {
		duration = g_pList.GetVectorData("SIN_DURATION").at(0);
		m_continuousMode = true;
	}
	else {
		duration = 1.0/freq;
		m_continuousMode = false;
	}

	if (g_pList.GetVectorData("SIN_MODE").at(0) == 0.0) {
		// Calculate the sinusoid for the platform.
		nmMovementData tmpTraj;
		double amp = g_pList.GetVectorData("SIN_TRANS_AMPLITUDE").at(0);
		double e = g_pList.GetVectorData("SIN_ELEVATION").at(0);
		double a = g_pList.GetVectorData("SIN_AZIMUTH").at(0);
		double step = 1.0/60.0;
		for (double i = 0.0; i < duration + step; i += step) {
			double val = amp*sin(2.0*PI*freq*i + 90.0*DEG2RAD);
			nm3DDatum cv = nmSpherical2Cartesian(e, a, val, true);
			tmpTraj.X.push_back(cv.x);
			tmpTraj.Y.push_back(cv.y);
#if DOF_MODE
			tmpTraj.Z.push_back(cv.z + MOTION_BASE_CENTER);
#else
			tmpTraj.Z.push_back(cv.z);
#endif
		}

		// Do the sinusoid for OpenGL.
		amp = g_pList.GetVectorData("SIN_TRANS_AMPLITUDE").at(1);
		e = g_pList.GetVectorData("SIN_ELEVATION").at(1);
		a = g_pList.GetVectorData("SIN_AZIMUTH").at(1);
		for (double i = 0.0; i < duration + step; i += step) {
			double val = amp*sin(2.0*PI*freq*i + 90.0*DEG2RAD);
			nm3DDatum cv = nmSpherical2Cartesian(e, a, val, true);
			m_glData.X.push_back(cv.x);
			m_glData.Y.push_back(cv.y);
#if DOF_MODE
			m_glData.Z.push_back(cv.z + MOTION_BASE_CENTER);
#else
			m_glData.Z.push_back(cv.z);
#endif
		}
		GeneratePredictedData();

		// Calculates the trajectory to move the platform to start position.
		DATA_FRAME startFrame;
		startFrame.lateral = tmpTraj.X.at(0); startFrame.surge = tmpTraj.Y.at(0); startFrame.heave = tmpTraj.Z.at(0)-MOTION_BASE_CENTER;
		startFrame.yaw = startFrame.pitch = startFrame.roll = 0.0;
		MovePlatform(&startFrame);

		m_recordOffset = static_cast<int>(m_data.X.size());

		// Add the translational sinusoid to the trajectory.
		for (int i = 0; i < static_cast<int>(tmpTraj.X.size()); i++) {
			m_data.X.push_back(tmpTraj.X.at(i));
			m_data.Y.push_back(tmpTraj.Y.at(i));
			m_data.Z.push_back(tmpTraj.Z.at(i));
			m_rotData.X.push_back(0.0);
			m_rotData.Y.push_back(0.0);
			m_rotData.Z.push_back(0.0);
		}
	} // if (g_pList.GetVectorData("SIN_MODE").at(0) == 0.0)
	else {
		vector<double> platformCenter = g_pList.GetVectorData("PLATFORM_CENTER"),
					   headCenter = g_pList.GetVectorData("HEAD_CENTER"),
					   origin = g_pList.GetVectorData("M_ORIGIN"),
					   rotationOffsets = g_pList.GetVectorData("ROT_CENTER_OFFSETS"),
					   eyeOffsets = g_pList.GetVectorData("EYE_OFFSETS");
		double amp = g_pList.GetVectorData("SIN_ROT_AMPLITUDE").at(0);
		double e = -g_pList.GetVectorData("SIN_ELEVATION").at(0);
		double a = g_pList.GetVectorData("SIN_AZIMUTH").at(0);
		double step = 1.0/60.0;

		
		// Create the rotation profile of the sin.
		vector<double> tmpRotTraj;
		double val;
		for (double i = 0.0; i < duration + step; i += step) {
			if (motionType == 6) // Step Velocity
				val = 0.0;
			else val = amp*sin(2.0*PI*freq*i + 90.0*DEG2RAD);
			tmpRotTraj.push_back(val);
		}

		// Point is the center of the platform, rotPoint is the subject's head.
		nm3DDatum point, rotPoint;
		point.x = platformCenter.at(0); 
		point.y = platformCenter.at(1);
		point.z = platformCenter.at(2);
		rotPoint.x = headCenter.at(0)/100.0 + CENTROID_OFFSET_X + origin.at(0) + rotationOffsets.at(0)/100.0;
		rotPoint.y = headCenter.at(1)/100.0 + CENTROID_OFFSET_Y + origin.at(1) + rotationOffsets.at(1)/100.0;
		rotPoint.z = headCenter.at(2)/100.0 + CENTROID_OFFSET_Z + origin.at(2) + rotationOffsets.at(2)/100.0;

		vector<double> tmpTraj;
		nmMovementData tmpData, tmpRotData;
		nmRotatePointAboutPoint(point, rotPoint, e, a, &tmpRotTraj, &tmpData, &tmpRotData, true, true);

#if MINI_MOOG_SYSTEM 
		for (int i = 0; i < static_cast<int>(tmpRotData.X.size()); i++) {
			// The rexroth system uses radians instead of degrees, so we must convert the
			// output of the rotation calculation.
			tmpRotData.X.at(i) *= DEG2RAD;		// Yaw
			tmpRotData.Y.at(i) *= DEG2RAD;		// Pitch
			tmpRotData.Z.at(i) *= DEG2RAD;		// Roll
		}
#endif

		// Calculates the trajectory to move the platform to start position.
		DATA_FRAME startFrame;
		startFrame.lateral = tmpData.X.at(0); startFrame.surge = tmpData.Y.at(0); startFrame.heave = -tmpData.Z.at(0);
		startFrame.yaw = tmpRotData.X.at(0); startFrame.pitch = tmpRotData.Y.at(0); startFrame.roll = -tmpRotData.Z.at(0);
		MovePlatform(&startFrame);

		m_recordOffset = static_cast<int>(m_data.X.size());

		// Add the rotational sinusoid to the trajectory.
		for (int i = 0; i < static_cast<int>(tmpData.X.size()); i++) {
			m_data.X.push_back(tmpData.X.at(i));
			m_data.Y.push_back(tmpData.Y.at(i));
			m_data.Z.push_back(-tmpData.Z.at(i) + MOTION_BASE_CENTER);
			m_rotData.X.push_back(tmpRotData.X.at(i));
			m_rotData.Y.push_back(tmpRotData.Y.at(i));
			m_rotData.Z.push_back(-tmpRotData.Z.at(i));
		}

		/*********** Johnny add for dawning OpenGL *************/

		// This tells Compute() to use the rotation info and the GLPanel to use
		// the rotation transformations in Render().
		m_setRotation = true;
		m_glWindow->GetGLPanel()->DoRotation(true);

		// have problem and not fix yet
		// Choose whether we're rotating the fixation point.
		bool rotFP = g_pList.GetVectorData("FP_ROTATE").at(0) >= 1.0 ? true : false;
		m_glWindow->GetGLPanel()->RotationType(static_cast<int>(g_pList.GetVectorData("FP_ROTATE").at(0)));

		double gl_amp = g_pList.GetVectorData("SIN_ROT_AMPLITUDE").at(1);
		double gl_elevation = -g_pList.GetVectorData("SIN_ELEVATION").at(1);
		double gl_azimuth = g_pList.GetVectorData("SIN_AZIMUTH").at(1);
		double gl_freq = g_pList.GetVectorData("SIN_FREQUENCY").at(1);

		// Create the rotation profile of the sin.
		vector<double> angleTrajectory;
		for (double i = 0.0; i < duration + step; i += step) {
			if (motionType == 6) // Step Velocity
				val = 2.0*PI*gl_freq*i*(180/PI); // degree
			else val = gl_amp*sin(2.0*PI*gl_freq*i + 90.0*DEG2RAD);
			angleTrajectory.push_back(val);
		}

#if USE_MATLAB
		stuffDoubleVector(angleTrajectory, "angleTrajectory");
#endif

		// Calculate the rotation vector describing the axis of rotation.
		m_rotationVector = nmSpherical2Cartesian(gl_elevation, gl_azimuth, 1.0, true);

		// Swap the y and z values of the rotation vector to accomodate OpenGL.  We also have
		// to negate the y value because forward is negative in our OpenGL axes.
		double tmp = -m_rotationVector.y;
		m_rotationVector.y = m_rotationVector.z;
		m_rotationVector.z = tmp;
		m_glWindow->GetGLPanel()->SetRotationVector(m_rotationVector);

		// Calulation of offsets for rotation.
		double xdist = -eyeOffsets.at(0) + rotationOffsets.at(3),
			ydist = -eyeOffsets.at(2) + rotationOffsets.at(5),
			zdist = CENTER2SCREEN - headCenter.at(1) - rotationOffsets.at(4);
		m_glWindow->GetGLPanel()->SetRotationCenter(xdist, ydist, zdist);

		// Fill up the OpenGL trajectories for both the rotation and translation.  Translation
		// is actually just set to the zero position because, for this type of movement, the
		// monkey's head shouldn't be translating.
		m_glRotData.clear();
		nmClearMovementData(&m_glData);
		for (int i = 0; i < static_cast<int>(angleTrajectory.size()); i++) {
			m_glRotData.push_back(-angleTrajectory.at(i)); 
			m_glData.X.push_back(0.0); m_glData.Y.push_back(0.0); m_glData.Z.push_back(MOTION_BASE_CENTER);
		}
		GeneratePredictedRotationData();
		GeneratePredictedData();

		// when we draw max and min amplitude line for calibration,
		// we setup the predicted min and max rotation angle.
		if (g_pList.GetVectorData("CALIB_ROT_ON").at(0) == 1.0){
			double minPredictedAngle = m_interpRotation.at(0);
			double maxPredictedAngle = m_interpRotation.at(0);
			double tmpAngle;
			int size = m_interpRotation.size();
			for(int i=5; i<size; i++){
				tmpAngle = m_interpRotation.at(i);
				if(tmpAngle < minPredictedAngle) minPredictedAngle = tmpAngle;
				if(tmpAngle > maxPredictedAngle) maxPredictedAngle = tmpAngle;
			}
			m_glWindow->GetGLPanel()->minPredictedAngle = minPredictedAngle;
			m_glWindow->GetGLPanel()->maxPredictedAngle = maxPredictedAngle;
		}

		// If we're rotatiting the fixation point, we need to generate a trajectory for it
		// so we can spit out values on the D to A board for Tempo to use.
		if (rotFP == true) {
			// Point that will be rotated.
			point.x = 0.0;
			point.y = CENTER2SCREEN - headCenter.at(1) - eyeOffsets.at(1);
			point.z = 0.0;

			rotPoint.x = xdist; rotPoint.y = point.y - zdist; rotPoint.z = ydist;
			double elevation = e, azimuth = a;
			nmRotatePointAboutPoint(point, rotPoint, elevation, azimuth, &angleTrajectory,
									&m_fpData, &m_fpRotData, true, true);

			for (int i = 0; i < static_cast<int>(m_fpData.X.size()); i++) {
				// Find the total distance traveled from the start to end point.
				double x = m_fpData.X.at(i),
					y = m_fpData.Y.at(i),
					z = m_fpData.Z.at(i);
				double tdist = sqrt(x*x + y*y + z*z) / 2.0;

				// If tdist is 0.0 then set elevation manually, otherwise we'll get a divide by zero
				// error.
				if (tdist != 0.0) {
					elevation = asin(z / (tdist*2.0));
				}
				else {
					elevation = 0.0;
				}

				azimuth = atan2(y, x);

				m_fpRotData.X.at(i) = azimuth*RAD2DEG - 90.0;
				m_fpRotData.Y.at(i) = -elevation*RAD2DEG;
			}
		}

	}
}

void MoogDotsCom::CalculateStepVelocityMovement()
{
	// This tells Compute() not to set any rotation info and the GLPanel not to try
	// to do any rotation transformations in Render().
	m_setRotation = false;
	m_glWindow->GetGLPanel()->DoRotation(false);

	// Make sure we don't rotate the fixation point.
	m_glWindow->GetGLPanel()->RotationType(0);

	// Do no move these initializations.  Their location in the function is very important for
	// threading issues.
	m_grabIndex = 0;
	m_recordIndex = 0;

	// Clear the OpenGL data.
	nmClearMovementData(&m_glData);

	m_continuousMode = true;

	// Get all step velocity movement parameters, it only works in visual.
	angularSpeed = g_pList.GetVectorData("STEP_VELOCITY_PARA").at(0);
	double duration = g_pList.GetVectorData("STEP_VELOCITY_PARA").at(1);
	double gl_azimuth = g_pList.GetVectorData("STEP_VELOCITY_PARA").at(2);
	double gl_elevation = -g_pList.GetVectorData("STEP_VELOCITY_PARA").at(3);

	vector<double> platformCenter = g_pList.GetVectorData("PLATFORM_CENTER"),
					headCenter = g_pList.GetVectorData("HEAD_CENTER"),
					origin = g_pList.GetVectorData("M_ORIGIN"),
					rotationOffsets = g_pList.GetVectorData("ROT_CENTER_OFFSETS"),
					eyeOffsets = g_pList.GetVectorData("EYE_OFFSETS");
	double step = 1.0/60.0;

		
	// Create the rotation profile for doing nothing on platform movement.
	vector<double> tmpRotTraj;
	for (double i = 0.0; i < duration + step; i += step) {
		tmpRotTraj.push_back(0.0);
	}

	// Point is the center of the platform, rotPoint is the subject's head.
	nm3DDatum point, rotPoint;
	point.x = platformCenter.at(0); 
	point.y = platformCenter.at(1);
	point.z = platformCenter.at(2);
	rotPoint.x = headCenter.at(0)/100.0 + CENTROID_OFFSET_X + origin.at(0) + rotationOffsets.at(0)/100.0;
	rotPoint.y = headCenter.at(1)/100.0 + CENTROID_OFFSET_Y + origin.at(1) + rotationOffsets.at(1)/100.0;
	rotPoint.z = headCenter.at(2)/100.0 + CENTROID_OFFSET_Z + origin.at(2) + rotationOffsets.at(2)/100.0;

	vector<double> tmpTraj;
	nmMovementData tmpData, tmpRotData;
	nmRotatePointAboutPoint(point, rotPoint, 0.0, 0.0, &tmpRotTraj, &tmpData, &tmpRotData, true, true);

	// Calculates the trajectory to move the platform to start position.
	DATA_FRAME startFrame;
	startFrame.lateral = tmpData.X.at(0); startFrame.surge = tmpData.Y.at(0); startFrame.heave = -tmpData.Z.at(0);
	startFrame.yaw = tmpRotData.X.at(0); startFrame.pitch = tmpRotData.Y.at(0); startFrame.roll = -tmpRotData.Z.at(0);
	MovePlatform(&startFrame);

	m_recordOffset = static_cast<int>(m_data.X.size());

	// Add the rotational sinusoid to the trajectory.
	for (int i = 0; i < static_cast<int>(tmpData.X.size()); i++) {
		m_data.X.push_back(tmpData.X.at(i));
		m_data.Y.push_back(tmpData.Y.at(i));
		m_data.Z.push_back(-tmpData.Z.at(i) + MOTION_BASE_CENTER);
		m_rotData.X.push_back(tmpRotData.X.at(i));
		m_rotData.Y.push_back(tmpRotData.Y.at(i));
		m_rotData.Z.push_back(-tmpRotData.Z.at(i));
	}

	/*********** Johnny add for dawning OpenGL *************/

	// This tells Compute() to use the rotation info and the GLPanel to use
	// the rotation transformations in Render().
	m_setRotation = true;
	m_glWindow->GetGLPanel()->DoRotation(true);

	// Create the rotation profile of the sin.
	vector<double> angleTrajectory;
	for (double i = 0.0; i < duration + step; i += step) {
		angleTrajectory.push_back(angularSpeed*i); //degree
	}

#if USE_MATLAB
	stuffDoubleVector(angleTrajectory, "angleTrajectory");
#endif

	// Calculate the rotation vector describing the axis of rotation.
	m_rotationVector = nmSpherical2Cartesian(gl_elevation, gl_azimuth, 1.0, true);

	// Swap the y and z values of the rotation vector to accomodate OpenGL.  We also have
	// to negate the y value because forward is negative in our OpenGL axes.
	double tmp = -m_rotationVector.y;
	m_rotationVector.y = m_rotationVector.z;
	m_rotationVector.z = tmp;
	m_glWindow->GetGLPanel()->SetRotationVector(m_rotationVector);

	// Calulation of offsets for rotation.
	double xdist = -eyeOffsets.at(0) + rotationOffsets.at(3),
		ydist = -eyeOffsets.at(2) + rotationOffsets.at(5),
		zdist = CENTER2SCREEN - headCenter.at(1) - rotationOffsets.at(4);
	m_glWindow->GetGLPanel()->SetRotationCenter(xdist, ydist, zdist);

	// Fill up the OpenGL trajectories for both the rotation and translation.  Translation
	// is actually just set to the zero position because, for this type of movement, the
	// monkey's head shouldn't be translating.
	m_glRotData.clear();
	nmClearMovementData(&m_glData);
	for (int i = 0; i < static_cast<int>(angleTrajectory.size()); i++) {
		m_glRotData.push_back(-angleTrajectory.at(i)); 
		m_glData.X.push_back(0.0); m_glData.Y.push_back(0.0); m_glData.Z.push_back(MOTION_BASE_CENTER);
	}
	GeneratePredictedRotationData();
	GeneratePredictedData();
}

void MoogDotsCom::CalculateTiltTranslationMovement()
{
	nmMovementData translationalData;
	int ttMode = g_pList.GetVectorData("TT_MODE").at(0);

	// If we're doing subtractive tilt/translation, then we need to add 90 degrees
	// to the axis of rotation to get the axis of translation.  If we're doing additive,
	// we need to subtract 90 degrees.
	double offset = 90.0;
	if (ttMode == 1.0 || ttMode == 4.0) {
		offset = -90.0;
	}

	// The axis of translation is 90 degrees away from the axis of rotation.
	double amplitude = g_pList.GetVectorData("ROT_AMPLITUDE").at(0),
		   gl_amplitude = g_pList.GetVectorData("ROT_AMPLITUDE").at(1),
		   azimuth = g_pList.GetVectorData("ROT_AZIMUTH").at(0) + offset,
		   gl_azimuth = g_pList.GetVectorData("ROT_AZIMUTH").at(1) + offset,
		   duration = g_pList.GetVectorData("ROT_DURATION").at(0),
		   gl_duration = g_pList.GetVectorData("ROT_DURATION").at(1),
		   sigma = g_pList.GetVectorData("ROT_SIGMA").at(0),
		   gl_sigma = g_pList.GetVectorData("ROT_SIGMA").at(1),
		   step = 1.0/60.0;

	// This does all the rotational stuff for us.  Then all we have to do
	// is calculate the translational offsets and add them in.  If we want a pure translation
	// movement, we set the amplitude to zero, call the function, then set amplitude back.
	// This effectively sets all our data in the trajectory to zero.
	if (ttMode == 3.0 || ttMode == 4.0) {
		vector<double> zeroVector, tmpVector = g_pList.GetVectorData("ROT_AMPLITUDE");
		zeroVector.push_back(0.0); zeroVector.push_back(gl_amplitude);
		g_pList.SetVectorData("ROT_AMPLITUDE", zeroVector);
		CalculateRotationMovement();
		g_pList.SetVectorData("ROT_AMPLITUDE", tmpVector);
	}
	else {
		CalculateRotationMovement();
	}

	// Make sure we don't rotate the fixation point.
	m_glWindow->GetGLPanel()->RotationType(0);

	// Generates a Gaussian that reflects the rotation amplitude over time.
	vector<double> tmpTraj;
	nmGen1DVGaussTrajectory(&tmpTraj, amplitude, duration, 60.0, sigma, 0.0, false);

	// Create a vector that holds all the values for the acceleration
	// we need to match.
	vector<double> taccel;
	for (int i = 0; i < static_cast<int>(tmpTraj.size()); i++) {
		taccel.push_back(sin(tmpTraj.at(i)*DEG2RAD)*9.82);
	}

	// Now integrate our acceleration vector twice to get our position vector.
	vector<double> v, p;
	double isum;
	nmTrapIntegrate(&taccel, &v, isum, 0, static_cast<int>(taccel.size())-1, step);
	nmTrapIntegrate(&v, &p, isum, 0, static_cast<int>(v.size())-1, step);

	// Now we need to rotate the position data given the specified azimuth.  It is assumed
	// that elevation will always be zero.  We only add the x and y components since z is
	// unused.
	double oX = g_pList.GetVectorData("M_ORIGIN").at(0),
		   oY = g_pList.GetVectorData("M_ORIGIN").at(1);
	for (int i = 0; i < static_cast<int>(p.size()); i++) {
		// This indexes us into the correct location in the movement data.
		int index = i + m_recordOffset;

		nm3DDatum d = nmSpherical2Cartesian(0.0, azimuth, p.at(i), true);

		// If we're doing translation movement only, we disregard previous values in the data so that
		// we only have a pure tilt/translation trajectory without rotation compensation
		// data.  If we're doing rotational movement only, we don't add in the translation
		// compensation for gravity.
		switch (ttMode)
		{
			// Rotation + translation
		case 0:
		case 1:
			m_data.X.at(index) += d.x;
			m_data.Y.at(index) += d.y;
			break;

			// Movement only
		case 3:
		case 4:
			m_data.X.at(index) = d.x + oX;
			m_data.Y.at(index) = d.y + oY;
			break;
		};
	}

	// We need to add a buffered stop to the end of the movement so it doesn't jerk, so
	// we employ the same method as when we want to make an emergency stop.  First, we'll
	// grab the last value of the translational and rotation data because we're going to
	// use it many times.
	int lastVal = static_cast<int>(m_data.X.size()) - 1,
		previousVal = lastVal - 1;
	double eLateral = m_data.X.at(lastVal),
		   eSurge = m_data.Y.at(lastVal),
		   eHeave = m_data.Z.at(lastVal),
		   eYaw = m_rotData.X.at(lastVal),
		   ePitch = m_rotData.Y.at(lastVal),
		   eRoll = m_rotData.Z.at(lastVal);
	
	// Calculate the instantaneous velocity for each axis.
	double ixv = eLateral - m_data.X.at(previousVal),
		   iyv = eSurge - m_data.Y.at(previousVal),
		   izv = eHeave - m_data.Z.at(previousVal),
		   iyawv = eYaw - m_rotData.X.at(previousVal),
		   ipitchv = ePitch - m_rotData.Y.at(previousVal),
		   irollv = eRoll - m_rotData.Z.at(previousVal);

	// Create buffered movement data.
	for (int i = 0; i < SPEED_BUFFER_SIZE; i++) {
		// Translational movement data.
		eLateral += ixv * m_speedBuffer2[i];
		eSurge += iyv * m_speedBuffer2[i];
		eHeave += izv * m_speedBuffer2[i];
		m_data.X.push_back(eLateral);
		m_data.Y.push_back(eSurge);
		m_data.Z.push_back(eHeave);

		// Rotational movement data.
		eYaw += iyawv * m_speedBuffer2[i];
		ePitch += ipitchv * m_speedBuffer2[i];
		eRoll += irollv * m_speedBuffer2[i];
		m_rotData.X.push_back(eYaw);
		m_rotData.Y.push_back(ePitch);
		m_rotData.Z.push_back(eRoll);
	}

#if DEBUG_DEFAULTS && USE_MATLAB
	stuffDoubleVector(tmpTraj, "rpos");
	stuffDoubleVector(taccel, "taccel");
	stuffDoubleVector(v, "tvel");
	stuffDoubleVector(p, "tpos");
#endif
}


void MoogDotsCom::CalculateRotationMovement()
{
	nmMovementData tmpData, tmpRotData;

	m_continuousMode = false;

	vector<double> platformCenter = g_pList.GetVectorData("PLATFORM_CENTER"),
		           headCenter = g_pList.GetVectorData("HEAD_CENTER"),
				   origin = g_pList.GetVectorData("M_ORIGIN"),
				   rotationOffsets = g_pList.GetVectorData("ROT_CENTER_OFFSETS"),
				   eyeOffsets = g_pList.GetVectorData("EYE_OFFSETS");

	// This tells Compute() to use the rotation info and the GLPanel to use
	// the rotation transformations in Render().
	m_setRotation = true;
	m_glWindow->GetGLPanel()->DoRotation(true);

	// Choose whether we're rotating the fixation point.
	bool rotFP = g_pList.GetVectorData("FP_ROTATE").at(0) >= 1.0 ? true : false;
	m_glWindow->GetGLPanel()->RotationType(static_cast<int>(g_pList.GetVectorData("FP_ROTATE").at(0)));

#if USE_MATLAB
	// Values that are only really used when taking debug and feedback data through Matlab.
#if RECORD_MODE
	m_recordedLateral.clear(); m_recordedHeave.clear(); m_recordedSurge.clear();
	m_recordedLateral.reserve(5000); m_recordedHeave.reserve(5000); m_recordedSurge.reserve(5000);
	m_recordedYaw.clear(); m_recordedPitch.clear(); m_recordedRoll.clear();
	m_recordedYaw.reserve(5000); m_recordedPitch.reserve(5000); m_recordedRoll.reserve(5000);
#endif
	m_sendStamp.clear(); m_receiveStamp.clear();
#endif

	// Do no move these initializations.  Their location in the function is very important for
	// threading issues.
	m_grabIndex = 0;
	m_recordIndex = 0;

	// Point is the center of the platform, rotPoint is the subject's head + offsets.
	nm3DDatum point, rotPoint;
	point.x = platformCenter.at(0) + origin.at(0); 
	point.y = platformCenter.at(1) + origin.at(1);
	point.z = platformCenter.at(2) + origin.at(2);
	rotPoint.x = headCenter.at(0)/100.0 + CENTROID_OFFSET_X + origin.at(0) + rotationOffsets.at(0)/100.0;
	rotPoint.y = headCenter.at(1)/100.0 + CENTROID_OFFSET_Y + origin.at(1) + rotationOffsets.at(1)/100.0;
	rotPoint.z = headCenter.at(2)/100.0 + CENTROID_OFFSET_Z + origin.at(2) + rotationOffsets.at(2)/100.0;

	// Parameters for the rotation.
	double amplitude = g_pList.GetVectorData("ROT_AMPLITUDE").at(0),
		   gl_amplitude = g_pList.GetVectorData("ROT_AMPLITUDE").at(1),
		   duration = g_pList.GetVectorData("ROT_DURATION").at(0),
		   gl_duration = g_pList.GetVectorData("ROT_DURATION").at(1),
		   sigma = g_pList.GetVectorData("ROT_SIGMA").at(0),
		   gl_sigma = g_pList.GetVectorData("ROT_SIGMA").at(1),

		   // We negate elevation to be consistent with previous program conventions.
		   elevation = -g_pList.GetVectorData("ROT_ELEVATION").at(0),
		   gl_elevation = -g_pList.GetVectorData("ROT_ELEVATION").at(1),

		   azimuth = g_pList.GetVectorData("ROT_AZIMUTH").at(0),
		   gl_azimuth = g_pList.GetVectorData("ROT_AZIMUTH").at(1),
		   step = 1.0/60.0;

	// Generate the rotation amplitude with a Gaussian velocity profile.
	vector<double> tmpTraj;
	nmGen1DVGaussTrajectory(&tmpTraj, amplitude, duration, 60.0, sigma, 0.0, false);
	nmRotatePointAboutPoint(point, rotPoint, elevation, azimuth, &tmpTraj,
							&tmpData, &tmpRotData, true, true);

	// Create the Gaussian rotation trajectory for the OpenGL side of things.
	vector<double> angleTrajectory;
	nmGen1DVGaussTrajectory(&angleTrajectory, gl_amplitude, gl_duration, 60.0, gl_sigma, 0.0, false);

	// Flip the sign of the roll because the Moog needs to the roll to be opposite of what
	// the preceding function generates.  We also flip the sign of the heave because the
	// equations assume positive is up, whereas the Moog thinks negative is up.
	for (int i = 0; i < static_cast<int>(tmpRotData.X.size()); i++) {
		// The rexroth system uses radians instead of degrees, so we must convert the
		// output of the rotation calculation.
#if MINI_MOOG_SYSTEM
		tmpRotData.X.at(i) *= DEG2RAD;		// Yaw
		tmpRotData.Y.at(i) *= DEG2RAD;		// Pitch
		tmpRotData.Z.at(i) *= -DEG2RAD;		// Roll
		tmpData.Z.at(i) *= -1.0;			// Heave
#else
		tmpRotData.Z.at(i) *= -1.0;			// Roll
		tmpData.Z.at(i) *= -1.0;			// Heave
#endif
	}

	// Calculate the rotation vector describing the axis of rotation.
	m_rotationVector = nmSpherical2Cartesian(gl_elevation, gl_azimuth, 1.0, true);

	// Swap the y and z values of the rotation vector to accomodate OpenGL.  We also have
	// to negate the y value because forward is negative in our OpenGL axes.
	double tmp = -m_rotationVector.y;
	m_rotationVector.y = m_rotationVector.z;
	m_rotationVector.z = tmp;
	m_glWindow->GetGLPanel()->SetRotationVector(m_rotationVector);

	// Calulation of offsets for rotation.
	double xdist = -eyeOffsets.at(0) + rotationOffsets.at(3),
		   ydist = -eyeOffsets.at(2) + rotationOffsets.at(5),
		   zdist = CENTER2SCREEN - headCenter.at(1) - rotationOffsets.at(4);
	m_glWindow->GetGLPanel()->SetRotationCenter(xdist, ydist, zdist);

	// Fill up the OpenGL trajectories for both the rotation and translation.  Translation
	// is actually just set to the zero position because, for this type of movement, the
	// monkey's head shouldn't be translating.
	m_glRotData.clear();
	nmClearMovementData(&m_glData);
	for (int i = 0; i < static_cast<int>(angleTrajectory.size()); i++) {
		m_glRotData.push_back(-angleTrajectory.at(i));
		m_glData.X.push_back(0.0); m_glData.Y.push_back(0.0); m_glData.Z.push_back(MOTION_BASE_CENTER);
	}
	GeneratePredictedRotationData();
	GeneratePredictedData();

	// If we're rotatiting the fixation point, we need to generate a trajectory for it
	// so we can spit out values on the D to A board for Tempo to use.
	if (rotFP == true) {
		// Point that will be rotated.
		point.x = 0.0;
		point.y = CENTER2SCREEN - headCenter.at(1) - eyeOffsets.at(1);
		point.z = 0.0;

		rotPoint.x = xdist; rotPoint.y = point.y - zdist; rotPoint.z = ydist;
		nmRotatePointAboutPoint(point, rotPoint, elevation, azimuth, &angleTrajectory,
								&m_fpData, &m_fpRotData, true, true);

		for (int i = 0; i < static_cast<int>(m_fpData.X.size()); i++) {
			// Find the total distance traveled from the start to end point.
			double x = m_fpData.X.at(i),
				   y = m_fpData.Y.at(i),
				   z = m_fpData.Z.at(i);
			double tdist = sqrt(x*x + y*y + z*z) / 2.0;

			// If tdist is 0.0 then set elevation manually, otherwise we'll get a divide by zero
			// error.
			if (tdist != 0.0) {
				elevation = asin(z / (tdist*2.0));
			}
			else {
				elevation = 0.0;
			}

			azimuth = atan2(y, x);

			m_fpRotData.X.at(i) = azimuth*RAD2DEG - 90.0;
			m_fpRotData.Y.at(i) = -elevation*RAD2DEG;
		}
	}


#if DEBUG_DEFAULTS && USE_MATLAB
	stuffDoubleVector(tmpData.X, "tx");
	stuffDoubleVector(tmpData.Y, "ty");
	stuffDoubleVector(tmpData.Z, "tz");
	stuffDoubleVector(tmpRotData.X, "trx");
	stuffDoubleVector(tmpRotData.Y, "tr_y");
	stuffDoubleVector(tmpRotData.Z, "trz");
	stuffDoubleVector(m_glRotData, "rotData");
	stuffDoubleVector(m_interpRotation, "irotData");
	stuffDoubleVector(m_fpData.X, "fpX");
	stuffDoubleVector(m_fpData.Y, "fpY");
	stuffDoubleVector(m_fpData.Z, "fpZ");
#endif

	// Move the platform into starting position.
	DATA_FRAME startFrame;
	startFrame.lateral = tmpData.X.at(0);
	startFrame.surge = tmpData.Y.at(0);
	startFrame.heave = tmpData.Z.at(0);
	startFrame.yaw = tmpRotData.X.at(0);
	startFrame.pitch = tmpRotData.Y.at(0);
	startFrame.roll = tmpRotData.Z.at(0);
	MovePlatform(&startFrame);

	m_recordOffset = m_data.X.size();

	// Copy all the stuff found the temp vectors to the end of the main data vectors.
	for (int i = 0; i < static_cast<int>(tmpData.X.size()); i++) {
		m_data.X.push_back(tmpData.X.at(i));
		m_data.Y.push_back(tmpData.Y.at(i));
#if DOF_MODE
		m_data.Z.push_back(tmpData.Z.at(i) + MOTION_BASE_CENTER);
#else
		m_data.Z.push_back(tmpData.Z.at(i));
#endif
		m_rotData.X.push_back(tmpRotData.X.at(i));
		m_rotData.Y.push_back(tmpRotData.Y.at(i));
		m_rotData.Z.push_back(tmpRotData.Z.at(i));
	}

//#if !CUSTOM_TIMER && !RECORD_MODE - Johnny
#if !RECORD_MODE
	if(!m_customTimer){
		m_delay = g_pList.GetVectorData("SYNC_DELAY").at(0);
		SyncNextFrame();
	}
#endif
}


//jw 5/26/20006
void MoogDotsCom::CalculateStimulusOutput()
{
	double magnitude = g_pList.GetVectorData("M_DIST").at(1);
	double duration = g_pList.GetVectorData("M_TIME").at(1);
	double sigma = g_pList.GetVectorData("M_SIGMA").at(1);

	// Calculate position trajectory for CED.
	nmGen1DVGaussTrajectory(&m_gaussianTrajectoryData, magnitude, duration, 60.0, sigma, 0.0, true);	
	m_interpStimulusData = DifferenceFunc(LATERAL_POLE, LATERAL_ZERO, Axis::Stimulus);
	nmGenDerivativeCurve(&m_StimulusData, &m_interpStimulusData, 1/60.0, true);

	// Normalize the output data.
	vector<double> normFactors = g_pList.GetVectorData("STIM_NORM_FACTORS");
	nmNormalizeVector(&m_StimulusData, normFactors.at(0), normFactors.at(1));
}


// jw add 05/26/2006
void MoogDotsCom::OutputStimulusCurve(int index)
{
	int ulStat;
	WORD DataValue;

	if(index >= static_cast<int>(m_StimulusData.size())) {
		ulStat = cbFromEngUnits(m_PCI_DIO48H_Object.DIO_board_num, BIP10VOLTS, 0.0, &DataValue);
		ulStat = cbAOut(m_PCI_DIO48H_Object.DIO_board_num,0, BIP10VOLTS, DataValue);	
	}
	else {
		float EngUnits;
		EngUnits = m_StimulusData[index];
		ulStat = cbFromEngUnits(m_PCI_DIO48H_Object.DIO_board_num, BIP10VOLTS, EngUnits, &DataValue);
		ulStat = cbAOut(m_PCI_DIO48H_Object.DIO_board_num,0, BIP10VOLTS, DataValue);	
	}
}


// jw add 05/26/2006
void MoogDotsCom::AddOverlap2Inerval(int mFirstEndIndex,int glFirstEndIndex,int overlapCnt)
{
	// calculate 2 platform overlap interval movement with 2 zero delay inerval
	int endIndex, Index2,Index1;
    
	Index1 = mFirstEndIndex-overlapCnt;
    Index2=mFirstEndIndex;
	
	for(Index1;Index1<=mFirstEndIndex;Index1++)
	{
		Index2++;
		m_data.X.at(Index1)=m_data.X.at(Index1)+m_data.X.at(Index2)-m_data.X.at(mFirstEndIndex) ;
		m_data.Y.at(Index1)=m_data.Y.at(Index1)+m_data.Y.at(Index2)-m_data.Y.at(mFirstEndIndex);
		m_data.Z.at(Index1)=m_data.Z.at(Index1)+m_data.Z.at(Index2)-m_data.Z.at(mFirstEndIndex);
	}
	//move non overlap 2nd interval part to position
    Index2++;
	endIndex = m_data.X.size()-1;
	Index1 =mFirstEndIndex;
	for(Index2;Index2<=endIndex;Index2++)
	{
        Index1++;	
		m_data.X.at(Index1)=m_data.X.at(Index2);
		m_data.Y.at(Index1)=m_data.Y.at(Index2);
		m_data.Z.at(Index1)=m_data.Z.at(Index2);
	}

	for(Index1;Index1<endIndex;Index1++)
	{
		m_data.X.pop_back();
		m_data.Y.pop_back();
		m_data.Z.pop_back();
	}
   // Add-glWindow Data
	Index1 = glFirstEndIndex-overlapCnt;
    Index2=glFirstEndIndex;
	
	for(Index1;Index1<=glFirstEndIndex;Index1++)
	{
		Index2++;
		m_glData.X.at(Index1)=m_glData.X.at(Index1)+m_glData.X.at(Index2)-m_glData.X.at(glFirstEndIndex) ;
		m_glData.Y.at(Index1)=m_glData.Y.at(Index1)+m_glData.Y.at(Index2)-m_glData.Y.at(glFirstEndIndex);
		m_glData.Z.at(Index1)=m_glData.Z.at(Index1)+m_glData.Z.at(Index2)-m_glData.Z.at(glFirstEndIndex);
	}
	//move non overlap 2nd interval part to position
    Index2++;
	endIndex = m_glData.X.size()-1;
	Index1 =glFirstEndIndex;
	for(Index2;Index2<=endIndex;Index2++)
	{
        Index1++;	
		m_glData.X.at(Index1)=m_glData.X.at(Index2);
		m_glData.Y.at(Index1)=m_glData.Y.at(Index2);
		m_glData.Z.at(Index1)=m_glData.Z.at(Index2);
	}
	for(Index1;Index1<endIndex;Index1++)
	{
		m_glData.X.pop_back();
		m_glData.Y.pop_back();
		m_glData.Z.pop_back();
	}
}


void MoogDotsCom::MovePlatform(DATA_FRAME *destination)
{
	// Empty the data vectors, which stores the trajectory data.
	nmClearMovementData(&m_data);
	nmClearMovementData(&m_rotData);

	// Get the positions currently in the command buffer.  We use the thread safe
	// version of GetAxesPositions() here because MovePlatform() is called from
	// both the main GUI thread and the communication thread.
	DATA_FRAME currentFrame;
	GET_DATA_FRAME(&currentFrame);

#if DOF_MODE
	// We assume that the heave value passed to us is based around zero.  We must add an offset
	// to that value to adjust for the Moog's inherent offset on the heave axis.
	destination->heave += MOTION_BASE_CENTER;
#endif

	// Check to see if the motion base's current position is the same as the startPosition.  If so,
	// we don't need to move the base into position.
	if (fabs(destination->lateral - currentFrame.lateral) > TINY_NUMBER ||
		fabs(destination->surge - currentFrame.surge) > TINY_NUMBER ||
		fabs(destination->heave - currentFrame.heave) > TINY_NUMBER)
	{
		// Move the platform from its current location to start position.
		nm3DDatum sp, ep;
		sp.x = currentFrame.lateral; sp.y = currentFrame.surge; sp.z = currentFrame.heave;
		ep.x = destination->lateral; ep.y = destination->surge; ep.z = destination->heave;
		nmGen3DVGaussTrajectory(&m_data, sp, ep, 2.0, 60.0, 3.0, false);
	}

	// Make sure that we're not rotated at all.
	if (fabs(destination->yaw - currentFrame.yaw) > TINY_NUMBER ||
		fabs(destination->pitch - currentFrame.pitch) > TINY_NUMBER ||
		fabs(destination->roll - currentFrame.roll) > TINY_NUMBER)
	{
		// Set the Yaw.
		nmGen1DVGaussTrajectory(&m_rotData.X, destination->yaw-currentFrame.yaw, 2.0, 60.0, 3.0, currentFrame.yaw, false);

		// Set the Pitch.
		nmGen1DVGaussTrajectory(&m_rotData.Y, destination->pitch-currentFrame.pitch, 2.0, 60.0, 3.0, currentFrame.pitch, false);

		// Set the Roll.
		nmGen1DVGaussTrajectory(&m_rotData.Z, destination->roll-currentFrame.roll, 2.0, 60.0, 3.0, currentFrame.roll, false);
	}

	// Now we make sure that data in m_data and data in m_rotData has the same length.
	if (m_data.X.size() > m_rotData.X.size()) {
		for (int i = 0; i < (int)m_data.X.size(); i++) {
			m_rotData.X.push_back(currentFrame.yaw);
			m_rotData.Y.push_back(currentFrame.pitch);
			m_rotData.Z.push_back(currentFrame.roll);
		}
	}
	else if (m_data.X.size() < m_rotData.X.size()) {
		for (int i = 0; i < (int)m_rotData.X.size(); i++) {
			m_data.X.push_back(currentFrame.lateral);
			m_data.Y.push_back(currentFrame.surge);
			m_data.Z.push_back(currentFrame.heave);
		}
	}
}


void MoogDotsCom::Calculate2IntervalMovement()
{
	int i;
	double div = 60.0;

	m_continuousMode = false;

	// This tells Compute() not to set any rotation info and the GLPanel not to try
	// to do any rotation transformations in Render().
	m_setRotation = false;
	m_glWindow->GetGLPanel()->DoRotation(false);

	// Make sure we don't rotate the fixation point.
	m_glWindow->GetGLPanel()->RotationType(0);

	// Do no move these initializations.  Their location in the function is very important for
	// threading issues.
	m_grabIndex = 0;
	m_recordIndex = 0;

	// Move the platform into starting position.
	DATA_FRAME startFrame;
	vector<double> o = g_pList.GetVectorData("M_ORIGIN");
	startFrame.lateral = o.at(0);
	startFrame.surge = o.at(1);
	startFrame.heave = o.at(2);
	startFrame.yaw = 0.0f;
	startFrame.pitch = 0.0f;
	startFrame.roll = 0.0f;
	MovePlatform(&startFrame);

	m_recordOffset = m_data.X.size();

	// Grab the parameters we need for the calculations.
	vector<double> elevation = g_pList.GetVectorData("2I_ELEVATION"),
				   azimuth = g_pList.GetVectorData("2I_AZIMUTH"),
				   sigma = g_pList.GetVectorData("2I_SIGMA"),
				   duration = g_pList.GetVectorData("2I_TIME"),
				   magnitude = g_pList.GetVectorData("2I_DIST");

	// Calculate the 1st movement for the platform.
	nm3DDatum offsets;
	offsets.x = startFrame.lateral; offsets.y = startFrame.surge; offsets.z = startFrame.heave;
	nmGen3DVGaussTrajectory(&m_data, elevation.at(0), azimuth.at(0), magnitude.at(0),
						   duration.at(0), div, sigma.at(0), offsets, true, false);

	// 1st movement for OpenGL.
	nmGen3DVGaussTrajectory(&m_glData, elevation.at(2), azimuth.at(2), magnitude.at(2),
						   duration.at(2), div, sigma.at(2), offsets, true, true);

	// Add in the delay for the platform.
	vector<double> delays = g_pList.GetVectorData("2I_DELAY");
	int delayCount = static_cast<int>(delays.at(0)*div);
	int endIndex = static_cast<int>(m_data.X.size())-1;
	double lastValueX = m_data.X.at(endIndex),
		   lastValueY = m_data.Y.at(endIndex),
		   lastValueZ = m_data.Z.at(endIndex);
	for (i = 0; i < delayCount; i++) {
		m_data.X.push_back(lastValueX);
		m_data.Y.push_back(lastValueY);
		m_data.Z.push_back(lastValueZ);
	}

	// Now add in the delay for OpenGL.
	delayCount = static_cast<int>(delays.at(1)*div);
	endIndex = static_cast<int>(m_glData.X.size())-1;
	lastValueX = m_glData.X.at(endIndex);
	lastValueY = m_glData.Y.at(endIndex);
	lastValueZ = m_glData.Z.at(endIndex);
	for (i = 0; i < delayCount; i++) {
		m_glData.X.push_back(lastValueX);
		m_glData.Y.push_back(lastValueY);
		m_glData.Z.push_back(lastValueZ);
	}

	// 2nd movement for the platform.
	endIndex = static_cast<int>(m_data.X.size())-1;
	offsets.x = m_data.X.at(endIndex); offsets.y = m_data.Y.at(endIndex); offsets.z = m_data.Z.at(endIndex);
	nmGen3DVGaussTrajectory(&m_data, elevation.at(1), azimuth.at(1), magnitude.at(1),
						   duration.at(1), div, sigma.at(1), offsets, true, false);

	// 2nd movement for OpenGL.
	endIndex = static_cast<int>(m_glData.X.size())-1;
	offsets.x = m_glData.X.at(endIndex); offsets.y = m_glData.Y.at(endIndex); offsets.z = m_glData.Z.at(endIndex);
	nmGen3DVGaussTrajectory(&m_glData, elevation.at(3), azimuth.at(3), magnitude.at(3),
						   duration.at(3), div, sigma.at(3), offsets, true, false);

	// Make sure the yaw, pitch, and roll components are filled with zeros since we don't
	// want those to move.
	for (i = m_recordOffset; i < static_cast<int>(m_data.X.size()); i++) {
		m_rotData.X.push_back(0.0);
		m_rotData.Y.push_back(0.0);
		m_rotData.Z.push_back(0.0);
	}
	// Creates interpolated, predicted data based on the command signal in m_glData.
	GeneratePredictedData();

//#if !CUSTOM_TIMER && !RECORD_MODE - Johnny
#if !RECORD_MODE
	if(!m_customTimer){
		m_delay = g_pList.GetVectorData("SYNC_DELAY").at(0);
		SyncNextFrame();
	}
#endif
}


void MoogDotsCom::CalculateGaussianMovement(DATA_FRAME *startFrame, double elevation,
											double azimuth, double magnitude, double duration,
											double sigma, bool doSecondMovement)
{
	int i;

	m_continuousMode = false;

	// This tells Compute() not to set any rotation info and the GLPanel not to try
	// to do any rotation transformations in Render().
	m_setRotation = false;
	m_glWindow->GetGLPanel()->DoRotation(false);

	// Make sure we don't rotate the fixation point.
	m_glWindow->GetGLPanel()->RotationType(0);

	// Do no move these initializations.  Their location in the function is very important for
	// threading issues.
	m_grabIndex = 0;
	m_recordIndex = 0;

	// Clear the noise data.
	nmClearMovementData(&m_noise);
	nmClearMovementData(&m_filteredNoise);

#if USE_MATLAB
	// Values that are only really used when taking debug and feedback data through Matlab.
#if RECORD_MODE
	m_recordedLateral.clear(); m_recordedHeave.clear(); m_recordedSurge.clear();
	m_recordedYaw.clear(); m_recordedPitch.clear(); m_recordedRoll.clear();
#endif
	m_sendStamp.clear(); m_receiveStamp.clear();
#endif

	// Moves the platform to start position.
	MovePlatform(startFrame);

	m_recordOffset = m_data.X.size();

	// Sometimes we just want to move the motion base to a space fixed location.  In this
	// case, we don't make a second movement.
	if (doSecondMovement) {
		// Generate the trajectory for the platform.
		nm3DDatum offset;
		offset.x = startFrame->lateral; offset.y = startFrame->surge; offset.z = startFrame->heave;
		nmGen3DVGaussTrajectory(&m_data, elevation, azimuth, magnitude, duration, 60.0, sigma, offset, true, false);

		// GL scene movement.
		elevation = g_pList.GetVectorData("M_ELEVATION").at(1);
		azimuth = g_pList.GetVectorData("M_AZIMUTH").at(1);
#if USING_FISHEYE
		if(azimuth==90.0)
			elevation +=m_glWindow->GetGLPanel()->adjusted_ele_offset;
#endif
		magnitude = g_pList.GetVectorData("M_DIST").at(1);
		duration = g_pList.GetVectorData("M_TIME").at(1);
		sigma = g_pList.GetVectorData("M_SIGMA").at(1);

		nmGen3DVGaussTrajectory(&m_glData, elevation, azimuth, magnitude, duration, 60.0, sigma, offset, true, true);

		// Add noise to the signal if flagged.
		if (g_pList.GetVectorData("USE_NOISE").at(0)) {
			vector<double> noiseMag = g_pList.GetVectorData("NOISE_MAGNITUDE");
			nm3DDatum mag;
			mag.x = noiseMag.at(0); mag.y = noiseMag.at(1); mag.z = noiseMag.at(2);

			// Generate the filtered noise that we'll add to the command buffer.
			nmGenerateFilteredNoise((long)g_pList.GetVectorData("GAUSSIAN_SEED").at(0),
									(int)m_data.X.size() - m_recordOffset,
									g_pList.GetVectorData("CUTOFF_FREQ").at(0),
									mag, 1, true, true,
									&m_noise, &m_filteredNoise);

			// This is a function from NumericalMethods that will rotate a data set.
			nmRotateDataYZ(&m_filteredNoise, g_pList.GetVectorData("NOISE_AZIMUTH").at(0),
						   g_pList.GetVectorData("NOISE_ELEVATION").at(0));

			// Add the noise to the command and visual feed.
			for (i = 0; i < static_cast<int>(m_filteredNoise.X.size()); i++) {
				int index = i + m_recordOffset;

				// Command
				m_data.X.at(index) += m_filteredNoise.X.at(i);
				m_data.Y.at(index) += m_filteredNoise.Y.at(i);
				m_data.Z.at(index) += m_filteredNoise.Z.at(i);

				// Visual
				m_glData.X.at(i) += m_filteredNoise.X.at(i);
				m_glData.Y.at(i) += m_filteredNoise.Y.at(i);
				m_glData.Z.at(i) += m_filteredNoise.Z.at(i);
			}
		}

		// Creates interpolated, predicted data based on the command signal in m_glData.
		GeneratePredictedData();

		// Make sure the yaw, pitch, and roll components are filled with zeros for the
		// 2nd part of the movement.
		for (i = m_recordOffset; i < static_cast<int>(m_data.X.size()); i++) {
			m_rotData.X.push_back(0.0);
			m_rotData.Y.push_back(0.0);
			m_rotData.Z.push_back(0.0);
		}
	} // End if (doSecondMovement)

//#if !CUSTOM_TIMER - Johnny 6/17/07
	if(!m_customTimer){
		m_delay = g_pList.GetVectorData("SYNC_DELAY").at(0);
		SyncNextFrame();
	}
//#endif
}

void MoogDotsCom::CalculateGaborMovement()
{
	m_continuousMode = false;

	// This tells Compute() not to set any rotation info and the GLPanel not to try
	// to do any rotation transformations in Render().
	m_setRotation = false;
	m_glWindow->GetGLPanel()->DoRotation(false);

	// Make sure we don't rotate the fixation point.
	m_glWindow->GetGLPanel()->RotationType(0);

	// Do no move these initializations.  Their location in the function is very important for
	// threading issues.
	m_grabIndex = 0;
	m_recordIndex = 0;

	// Clear the OpenGL data.
	nmClearMovementData(&m_glData);

	/******************* calculate gabor trajectory of platform ********************/
	nmMovementData tmpTraj;
	vector<double> gaborTraj;
	double elevation, azimuth, amp, duration, sigma, freq, cycle, step, t0, gauss,max;

	// use Gaussian parameters
	elevation = g_pList.GetVectorData("M_ELEVATION").at(0);
	azimuth = g_pList.GetVectorData("M_AZIMUTH").at(0);
//#if USING_FISHEYE
//		if(azimuth!=90.0)
//			elevation +=m_moogCom->m_glWindow->GetGLPanel()->adjusted_ele_offset;
//#endif
	duration = g_pList.GetVectorData("M_TIME").at(0);
	sigma = g_pList.GetVectorData("M_SIGMA").at(0);
	// use Sinusoid parameters
	amp = g_pList.GetVectorData("SIN_TRANS_AMPLITUDE").at(0);
	freq = g_pList.GetVectorData("SIN_FREQUENCY").at(0);
	cycle = duration*freq;
	step = 1.0/60.0;
	t0 = duration/2.0;
	max = 0;

	for (double i = 0.0; i < duration + step; i += step) {
		double val = sin((i-t0)/duration*cycle* 2.0*PI);
		gauss = exp(-sqrt(2.0)*pow((i-t0)/(duration/sigma),2));
		gaborTraj.push_back(val*gauss);
		if (val*gauss > max) max = val*gauss;
	}

	int s = gaborTraj.size();
	for (int j = 0; j < s; j++) {
		// normalize to magnitude and find x, y and z component
		nm3DDatum cv = nmSpherical2Cartesian(elevation, azimuth, amp*gaborTraj.at(j)/max, true);
		tmpTraj.X.push_back(cv.x);
		tmpTraj.Y.push_back(cv.y);
#if DOF_MODE
		tmpTraj.Z.push_back(cv.z + MOTION_BASE_CENTER);
#else
		tmpTraj.Z.push_back(cv.z);
#endif
	}

#if USE_MATLAB
	stuffDoubleVector(tmpTraj.X, "tx");
	stuffDoubleVector(tmpTraj.Y, "ty");
	stuffDoubleVector(tmpTraj.Z, "tz");
#endif
	// Calculates the trajectory to move the platform to start position.
	DATA_FRAME startFrame;
	startFrame.lateral = tmpTraj.X.at(0); startFrame.surge = tmpTraj.Y.at(0); startFrame.heave = tmpTraj.Z.at(0)-MOTION_BASE_CENTER;
	startFrame.yaw = startFrame.pitch = startFrame.roll = 0.0;
	MovePlatform(&startFrame);

	m_recordOffset = static_cast<int>(m_data.X.size());

	// Add the gabor to the trajectory.
	for (int i = 0; i < static_cast<int>(tmpTraj.X.size()); i++) {
		m_data.X.push_back(tmpTraj.X.at(i));
		m_data.Y.push_back(tmpTraj.Y.at(i));
		m_data.Z.push_back(tmpTraj.Z.at(i));
		m_rotData.X.push_back(0.0);
		m_rotData.Y.push_back(0.0);
		m_rotData.Z.push_back(0.0);
	}

#if USE_MATLAB
	stuffDoubleVector(m_data.X, "dx");
	stuffDoubleVector(m_data.Y, "dy");
	stuffDoubleVector(m_data.Z, "dz");
#endif

	/*************************** Do the Gabor for OpenGL. ************************/
	// use Gaussian parameters
	elevation = g_pList.GetVectorData("M_ELEVATION").at(1);
	azimuth = g_pList.GetVectorData("M_AZIMUTH").at(1);
#if USING_FISHEYE
		if(azimuth==90.0)
			elevation +=m_glWindow->GetGLPanel()->adjusted_ele_offset;
#endif
	duration = g_pList.GetVectorData("M_TIME").at(1);
	sigma = g_pList.GetVectorData("M_SIGMA").at(1);
	// use Sinusoid parameters
	amp = g_pList.GetVectorData("SIN_TRANS_AMPLITUDE").at(1);
	freq = g_pList.GetVectorData("SIN_FREQUENCY").at(1);
	cycle = duration*freq;
	t0 = duration/2.0;
	max=0;

	gaborTraj.clear();
	for (double i = 0.0; i < duration + step; i += step) {
		double val = sin((i-t0)/duration*cycle* 2.0*PI);
		gauss = exp(-sqrt(2.0)*pow((i-t0)/(duration/sigma),2));
		gaborTraj.push_back(val*gauss);
		if (val*gauss > max) max = val*gauss;
	}

	s = gaborTraj.size();
	for (int j = 0; j < s; j++) {
		// normalize to magnitude and find x, y and z component
		nm3DDatum cv = nmSpherical2Cartesian(elevation, azimuth, amp*gaborTraj.at(j)/max, true);
		m_glData.X.push_back(cv.x);
		m_glData.Y.push_back(cv.y);
#if DOF_MODE
		m_glData.Z.push_back(cv.z + MOTION_BASE_CENTER);
#else
		m_glData.Z.push_back(cv.z);
#endif
	}

	GeneratePredictedData();

#if USE_MATLAB
	stuffDoubleVector(m_glData.X, "gx");
	stuffDoubleVector(m_glData.Y, "gy");
	stuffDoubleVector(m_glData.Z, "gz");
#endif

}

void MoogDotsCom::GeneratePredictedData()
{
	// Clear the data structures which hold the interpolated return feedback.
	m_interpHeave.clear(); m_interpSurge.clear(); m_interpLateral.clear();
	m_accelerLateral.clear(); m_accelerHeave.clear(); m_accelerSurge.clear();

	m_interpLateral = DifferenceFunc(LATERAL_POLE, LATERAL_ZERO, Axis::Lateral);
	m_accelerLateral = m_accelerData;
	m_interpHeave = DifferenceFunc(HEAVE_POLE, HEAVE_ZERO, Axis::Heave);
	m_accelerHeave = m_accelerData;
	m_interpSurge = DifferenceFunc(SURGE_POLE, SURGE_ZERO, Axis::Surge);
	m_accelerSurge = m_accelerData;
}


void MoogDotsCom::GeneratePredictedRotationData()
{
	int len;
	double *ypred, *tmppred, *i_ypred;

	double poleTerm = 0.1043*2.0*60.0,
		   zeroTerm = 0.0373*2.0*60.0;

	// Clear the rotation data.
	m_interpRotation.clear();

	len = static_cast<int>(m_glRotData.size());
	ypred = new double[len];

	// Since, as of this time, we have no transfer function for rotation,
	// we'll simply copy m_glRotData into ypred.
	for (int i = 0; i < len; i++) {
		ypred[i] = m_glRotData.at(i);
	}

	// Initialize the first value.
	ypred[0] = m_glRotData.at(0);

	// Here's Johnny...
	for (i = 1; i < len; i++) {
		ypred[i] = (1/(1+poleTerm)) * (-ypred[i-1]*(1-poleTerm) + m_glRotData.at(i)*(1+zeroTerm) +
					m_glRotData.at(i-1)*(1-zeroTerm));
	}

	// Interpolate the data.
	int interp_len, pred_len;
	tmppred = linear_interp(ypred, len, 10, interp_len);
	pred_len = static_cast<int>(500.0/16.667*10.0);
	i_ypred = new double[interp_len + pred_len];

	// Pad the interpolated data.
	for (i = 0; i < pred_len; i++) {
		i_ypred[i] = tmppred[0];
	}
	for (i = pred_len; i < pred_len + interp_len; i++) {
		i_ypred[i] = tmppred[i-pred_len];
	}

	// Stuff the interpolated data into a vector.
	int offset = static_cast<int>(g_pList.GetVectorData("PRED_OFFSET").at(0) / 16.667 * 10.0);
	for (i = offset; i < interp_len + pred_len; i += 10) {
		m_interpRotation.push_back(i_ypred[i]);
	}

	// Simulate accelerometer signal and then output by cbAOut() - Johnny 5/2/07
	m_accelerRotation.clear();
	int frame_delay = static_cast<int>(g_pList.GetVectorData("FRAME_DELAY")[0] / 16.667 * 10.0);
	offset = offset - frame_delay;
	for (i = offset; i < interp_len + pred_len; i += 10) {
		m_accelerRotation.push_back(i_ypred[i]);
	}

	// Delete dynamically allocated objects.
	delete [] tmppred;
	delete [] ypred;
	delete [] i_ypred;
}


vector<double> MoogDotsCom::DifferenceFunc(double pole, double zero, Axis axis)
{
	int i, len;
	double poleTerm, zeroTerm;
	double *ypred, *tmppred, *i_ypred;
	vector<double> x, interpData;
	string title;
	bool isHeave = false;

	poleTerm = pole*2.0*60.0;
	zeroTerm = zero*2.0*60.0;

	switch (axis)
	{
	case Axis::Heave:
		x = m_glData.Z;
		isHeave = true;
		title = "predHeave";
		break;
	case Axis::Lateral:
		x = m_glData.X;
		title = "predLateral";
		break;
	case Axis::Surge:
		x = m_glData.Y;
		title = "predSurge";
		break;
	case Axis::Stimulus:
		x = m_gaussianTrajectoryData;
		title = "predCED";
		break;
	};

	len = static_cast<int>(x.size());
	ypred = new double[len];

	// Initialize the first value.
	ypred[0] = x[0];

	// Here's Johnny...
	for (i = 1; i < len; i++) {
		ypred[i] = (1/(1+poleTerm)) * (-ypred[i-1]*(1-poleTerm) + x[i]*(1+zeroTerm) +
					x[i-1]*(1-zeroTerm));
	}

#if USE_MATLAB
	vector<double> test;
	for (i = 0; i < len; i++) {
		test.push_back(ypred[i]);
	}
	stuffDoubleVector(test, title.c_str());
#endif

	// Interpolate the data.
	int interp_len, pred_len;
	tmppred = linear_interp(ypred, len, 10, interp_len);
	pred_len = static_cast<int>(500.0/16.667*10.0);
	i_ypred = new double[interp_len + pred_len];

	// Pad the interpolated data.
	for (i = 0; i < pred_len; i++) {
		i_ypred[i] = tmppred[0];
	}
	for (i = pred_len; i < pred_len + interp_len; i++) {
		i_ypred[i] = tmppred[i-pred_len];
	}

	// Stuff the interpolated data into a vector.
	int offset = static_cast<int>(g_pList.GetVectorData("PRED_OFFSET")[0] / 16.667 * 10.0);
	if(axis != Axis::Stimulus) {
	   offset = static_cast<int>(g_pList.GetVectorData("PRED_OFFSET")[0] / 16.667 * 10.0);
	}
	else {
      offset = static_cast<int>(g_pList.GetVectorData("PRED_OFFSET_STIMULUS")[0] / 16.667 * 10.0);
	}
	for (i = offset; i < interp_len + pred_len; i += 10) {
		interpData.push_back(i_ypred[i]);
	}

	// Simulate accelerometer signal and then output by cbAOut() - Johnny 5/2/07
	m_accelerData.clear();
	int frame_delay = static_cast<int>(g_pList.GetVectorData("FRAME_DELAY")[0] / 16.667 * 10.0);
	offset = offset - frame_delay;
	for (i = offset; i < interp_len + pred_len; i += 10) {
		m_accelerData.push_back(i_ypred[i]);
	}

/* for test only
#if USE_MATLAB
	vector<double> test2;
	for (i = 0; i < pred_len + interp_len; i++) {
		test2.push_back(i_ypred[i]);
	}
	stuffDoubleVector(test2, "i_ypred");

	test2.clear();
	for (i = 0; i < (int)interpData.size(); i++) {
		test2.push_back(interpData.at(i));
	}
	stuffDoubleVector(test2, "interpData");

	test2.clear();
	for (i = 0; i < (int)m_accelerData.size(); i++) {
		test2.push_back(m_accelerData.at(i));
	}
	stuffDoubleVector(test2, "m_accelerData");
#endif
*/
	// Delete dynamically allocated objects.
	delete [] tmppred;
	delete [] ypred;
	delete [] i_ypred;

	return interpData;
}


vector<double> MoogDotsCom::convertPolar2Vector(double elevation, double azimuth, double magnitude)
{
	vector<double> convertedVector;
	double x, y, z;

	// Calculate the z-component.
	z = magnitude * sin(elevation);

	// Calculate the y-component.
	y = magnitude * cos(elevation) * sin(azimuth);

	// Calculate the x-componenet.
	x = magnitude * cos(elevation) * cos(azimuth);

	// Stuff the results into a vector.
	convertedVector.push_back(x);
	convertedVector.push_back(y);
	convertedVector.push_back(z);

	return convertedVector;
}


double MoogDotsCom::deg2rad(double deg)
{
	return deg / 180 * PI;
}

void MoogDotsCom::ConnectPipes()
{
	// If we're using pipes, wait for a connection.
	if (m_connectionType == MoogDotsCom::ConnectionType::Pipes) {
		ConnectNamedPipe(m_pipeHandle, &m_overlappedEvent);

		// Wait for the pipe to get signaled.
		//m_messageConsole->InsertItems(1, &wxString("Waiting for client connection..."), 0);
		wxBusyInfo wait("Waiting for client connection...");
		WaitForSingleObject(m_overlappedEvent.hEvent, INFINITE);
		m_messageConsole->InsertItems(1, &wxString("Connected Pipes!"), 0);

		// Check the result.
		DWORD junk;
		if (GetOverlappedResult(m_pipeHandle, &m_overlappedEvent, &junk, FALSE) == 0) {
			wxMessageDialog d(NULL, "GetOverlappedResult failed.");
			d.ShowModal();
		}

		ResetEvent(m_overlappedEvent.hEvent);
	}
}


