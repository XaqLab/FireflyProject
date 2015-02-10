#pragma once

#include "GlobalDefs.h"
#include "MainPanel.h"
#include "MoogDotsCom.h"
#include <wx/textfile.h>

// Describes the main windows which holds all the components for the
// application.
class CMainFrame : public wxFrame
{
private:
	wxMenuBar *m_menuBar;
	wxMenu *m_fileMenu,
		   *m_toolsMenu,
		   *m_connectionTypeMenu;
	CMainPanel *m_mainPanel;
	MoogDotsCom *m_moogCom;
	bool m_isRDXOpen;
	wxStatusBar *m_statBar;
	
	// Widget ID's
	enum
	{
		MENU_FILE_EXIT,
		MENU_TOOLS_DISCONNECT,
		MENU_TOOLS_CONNECT_PIPES,
		MENU_TOOLS_DISCONNECT_PIPES,
		MENU_TOOLS_TEMPO,
		MENU_TOOLS_TIMER,
		MENU_TOOLS_VERBOSE,
		MENU_TOOLS_VSYNC,
		MENU_SET_PACKET_RATE,
		MENU_TOOLS_CTYPE_TEMPO,
		MENU_TOOLS_CTYPE_SPIKE2,
		MENU_TOOLS_CTYPE_ANALOG
	};

public:
	// Default Constructor
	CMainFrame(const wxChar *title, int xpos, int ypos, int width, int height);

	// Exits the program.
	void OnMenuFileExit(wxCommandEvent &event);

	// Connects to the MBC and Tempo or Pipes.
	void CMainFrame::ConnectMBC();

	// Disconnects from the MBC.
	void OnMenuToolsDisconnect(wxCommandEvent &event);

	// Connects to the Pipes only.
	void OnMenuToolsConnectPipes(wxCommandEvent &event);

	// Disconnects from the Pipes only.
	void OnMenuToolsDisconnectPipes(wxCommandEvent &event);

	// Toggles Tempo control mode.
	void OnMenuToolsTempo(wxCommandEvent &event);

	// Toggles the use of the built-in timer for the thread.
	void OnMenuToolsTimer(wxCommandEvent &event);

	// Toggles verbose mode for the message console.
	void OnMenuToolsVerboseMode(wxCommandEvent &event);

	// Opens a dialog to set the packet rate for the communication thread.
	void OnMenuToolsPacketRate(wxCommandEvent &event);

	// Sets the communication listen mode to Tempo.
	void OnMenuToolsCTypeTempo(wxCommandEvent &event);

	// Sets the communication listen mode to Spike2.
	void OnMenuToolsCTypeSpike2(wxCommandEvent &event);

	// Sets the communication listen mode to Analog.
	void OnMenuToolsCTypeAnalog(wxCommandEvent &event);

	// Toggles syncing to the vertical refresh.
	void OnMenuToolsVSync(wxCommandEvent &event);

	void OnFrameClose(wxCloseEvent &event);

	~CMainFrame();

private:
	DECLARE_EVENT_TABLE()
};

// The Application
class MoogDots : public wxGLApp
{
private:
	CMainFrame *m_mainFrame;

public:
	virtual bool OnInit();
	void InitRig();
};

DECLARE_APP(MoogDots)
