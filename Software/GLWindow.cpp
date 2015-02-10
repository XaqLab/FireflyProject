#include "StdAfx.h"
#include "GLWindow.h"

// Parameter list -- Original declaration can be found in ParameterList.cpp
extern CParameterList g_pList;

extern int RIG_ROOM;
extern int SCREEN_WIDTH;
extern int SCREEN_HEIGHT;
extern float CENTER2SCREEN;

/****************************************************************************************/
/*	GLWindow Definitions ****************************************************************/
/****************************************************************************************/
GLWindow::GLWindow(const wxChar *title, int xpos, int ypos, int width, int height,
				   Frustum frustum, StarField starfield) :
wxFrame((wxFrame *) NULL, -1, title, wxPoint(xpos, ypos), wxSize(width, height), wxSIMPLE_BORDER)
{
	GetClientSize(&m_clientX, &m_clientY);

	// Setup the pixel format descriptor.
#if USE_STEREO
	int attribList[6];
	attribList[0] = WX_GL_STEREO;
	attribList[1] = WX_GL_DOUBLEBUFFER;
	attribList[2] = WX_GL_RGBA;
	attribList[3] = WX_GL_STENCIL_SIZE; attribList[4] = 8;
	attribList[5] = 0;
#else
	int attribList[5];
	attribList[0] = WX_GL_DOUBLEBUFFER;
	attribList[1] = WX_GL_RGBA;
	attribList[2] = WX_GL_STENCIL_SIZE; attribList[3] = 8;
	attribList[4] = 0;
#endif

	// Create the embedded panel where all the OpenGL stuff will be shown.
	m_glpanel = new GLPanel(this, m_clientX, m_clientY, frustum, starfield, attribList);
}


/****************************************************************************************/
/*	GLPanel Definitions *****************************************************************/
/****************************************************************************************/
BEGIN_EVENT_TABLE(GLPanel, wxGLCanvas)
EVT_PAINT(GLPanel::OnPaint)
EVT_SIZE(GLPanel::OnSize)
END_EVENT_TABLE()


GLPanel::GLPanel(wxWindow *parent, int width, int height, Frustum frustum, StarField starfield, int *attribList) :
wxGLCanvas(parent, -1, wxPoint(0, 0), wxSize(width, height), 0, "GLCanvas", attribList),
m_frustum(frustum), m_starfield(starfield), m_Heave(0.0), m_Surge(0.0), m_Lateral(0.0),
m_starArray(NULL), m_frameCount(1), starFieldVertex3D(NULL), redrawCallList(true)
{
	m_starFieldCallList = glGenLists(1);

	InitGL();
	GenerateStarField();

	// Rotation defaults.
	m_rotationAngle = 0.0;
	m_rotationVector.x = 0.0; m_rotationVector.y = 0.0; m_rotationVector.z = 0.0;
	m_doRotation = false;
	m_rotationType = 0;
	minPredictedAngle = 0.0; maxPredictedAngle = 0.0;

	SetupParameters();
}

void GLPanel::SetupParameters()
{
	adjusted_offset=g_pList.GetVectorData("ADJUSTED_OFFSET").at(0);
    adjusted_ele_offset=g_pList.GetVectorData("ADJUSTED_OFFSET").at(1);
	lineWidth = g_pList.GetVectorData("CALIB_LINE_SETUP").at(0);
	lineLength = g_pList.GetVectorData("CALIB_LINE_SETUP").at(1);
	calibTranOn[0]=g_pList.GetVectorData("CALIB_TRAN_ON").at(0); 
	calibTranOn[1]=g_pList.GetVectorData("CALIB_TRAN_ON").at(1); 
	calibTranOn[2]=g_pList.GetVectorData("CALIB_TRAN_ON").at(2); 
	calibTranOn[3]=g_pList.GetVectorData("CALIB_TRAN_ON").at(3); 
	calibTranOn[4]=g_pList.GetVectorData("CALIB_TRAN_ON").at(4); 
	calibRotOn[0]=g_pList.GetVectorData("CALIB_ROT_ON").at(0); 
	calibRotOn[1]=g_pList.GetVectorData("CALIB_ROT_ON").at(1);
	calibRotMotion=g_pList.GetVectorData("CALIB_ROT_MOTION").at(0);
	calibSquareOn = g_pList.GetVectorData("CALIB_SQUARE").at(0);
	squareSize = g_pList.GetVectorData("CALIB_SQUARE").at(1); //cm
	squareCenter[0] = g_pList.GetVectorData("CALIB_SQUARE").at(2);  
	squareCenter[1] = g_pList.GetVectorData("CALIB_SQUARE").at(3); 
	FP_cross[0] = g_pList.GetVectorData("FP_CROSS").at(0); 
	FP_cross[1] = g_pList.GetVectorData("FP_CROSS").at(1); 
	FP_cross[2] = g_pList.GetVectorData("FP_CROSS").at(2); 
	FP_cross[3] = g_pList.GetVectorData("FP_CROSS").at(3); 
	FP_cross[4] = g_pList.GetVectorData("FP_CROSS").at(4); 
	FP_cross[5] = g_pList.GetVectorData("FP_CROSS").at(5); 
	hollowSphereRadius = g_pList.GetVectorData("DOTS_SPHERE_PARA").at(0);

	//starFieldVertex3D = NULL;	

}

GLPanel::~GLPanel()
{
	delete m_starArray;

	if(starFieldVertex3D != NULL){
		delete [] starFieldVertex3D;
	}
}


GLvoid GLPanel::Render()
{
	double targOffset = 0.0;

	// If star lifetime is up and we flagged the use of star lifetime, then modify some of
	// the stars.
	if (m_frameCount++ % m_starfield.lifetime == 0 && m_starfield.use_lifetime == 1.0) {
		ModifyStarField();
	}

	////**** BACK LEFT BUFFER. ****//
#if USE_STEREO
	glDrawBuffer(GL_BACK_LEFT);
#else
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
#endif
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);		// Clears the current scene.

#if USING_FISHEYE
    double scrWidth=SCRHeight*atan(1.0)*4.0;
    double scrHeight=SCRHeight*atan(1.0);
    CalculateStereoFrustum(scrWidth,scrHeight, m_frustum.camera2screenDist,
		m_frustum.clipNear, m_frustum.clipFar, -m_frustum.eyeSeparation / 2.0f,
		m_frustum.worldOffsetX, m_frustum.worldOffsetZ);
#else
	// Setup the projection matrix.
	CalculateStereoFrustum(m_frustum.screenWidth, m_frustum.screenHeight, m_frustum.camera2screenDist,
		m_frustum.clipNear, m_frustum.clipFar, -m_frustum.eyeSeparation / 2.0f,
		m_frustum.worldOffsetX, m_frustum.worldOffsetZ);
#endif
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

#if USING_FISHEYE
// Setup the camera.
	gluLookAt(-m_frustum.eyeSeparation/2.0f+m_Lateral, 0.0f-m_Heave+DY+adjusted_offset, m_frustum.camera2screenDist-m_Surge,-m_frustum.eyeSeparation/2.0f+m_Lateral, 0.0f-m_Heave+DY+adjusted_offset, m_frustum.camera2screenDist-m_Surge-1.0f,0.0, 1.0, 0.0); // Which way is up
#else
	// Setup the camera.
	gluLookAt(-m_frustum.eyeSeparation/2.0f+m_Lateral, 0.0f-m_Heave, m_frustum.camera2screenDist-m_Surge,		// Camera origin
		-m_frustum.eyeSeparation/2.0f+m_Lateral, 0.0f-m_Heave, m_frustum.camera2screenDist-m_Surge-1.0f,	// Camera direction
		0.0, 1.0, 0.0); // Which way is up
#endif

	// If we are using the cutout, we need to setup the stencil buffer.
	if (m_starfield.useCutout == true) {
		// Turn off polygon smoothing otherwise we get weird lines in the
		// triangle fan.
		glDisable(GL_POLYGON_SMOOTH);

		// Use 0 for clear stencil, enable stencil test
		glClearStencil(0);
		glEnable(GL_STENCIL_TEST);

		// All drawing commands fail the stencil test, and are not
		// drawn, but increment the value in the stencil buffer.
		glStencilFunc(GL_NEVER, 0x0, 0x0);
		glStencilOp(GL_INCR, GL_INCR, GL_INCR);

		// Draw a circle.
		glColor3d(1.0, 1.0, 1.0);
		glBegin(GL_TRIANGLE_FAN);
		//glVertex3d(m_starfield.fixationPointLocation[0] + m_Lateral,
		//		   m_starfield.fixationPointLocation[1] - m_Heave,
		//		   );
		for(double dAngle = 0; dAngle <= 360.0; dAngle += 2.0) {
			glVertex3d(m_starfield.cutoutRadius * cos(dAngle*DEG2RAD) + m_starfield.fixationPointLocation[0] + m_Lateral,
				m_starfield.cutoutRadius * sin(dAngle*DEG2RAD) + m_starfield.fixationPointLocation[1] - m_Heave,
				m_starfield.fixationPointLocation[2] - m_Surge);
		}
		glEnd();

		// Now, allow drawing, except where the stencil pattern is 0x1
		// and do not make any further changes to the stencil buffer
		glStencilFunc(GL_NOTEQUAL, 0x1, 0x1);
		glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

		// Turn smoothing back on to draw the star field.
		glEnable(GL_POLYGON_SMOOTH);
	}
	else {
		glDisable(GL_STENCIL_TEST);
	}

	// Create the center dot.  This is not the same as the fixation target.
	// This is mainly used to calibrate movement.
	glTranslated(calibTranOn[2],calibTranOn[3],calibTranOn[4]);
	glLineWidth(lineWidth);
	glColor3d(1.0, 0.0, 0.0);
	glBegin(GL_LINES);
	// horizontal line
	if(calibTranOn[0]== 1){
		glVertex3d(lineLength/2, 0.0, 0.0);
		glVertex3d(-lineLength/2, 0.0, 0.0);
	}
	// vertical line
	if(calibTranOn[1]== 1){
		glVertex3d(0.0, lineLength/2, 0.0);
		glVertex3d(0.0, -lineLength/2, 0.0);
	}
	glEnd();
	glTranslated(-calibTranOn[2],-calibTranOn[3],-calibTranOn[4]);

	// Drawing a square for calibration
	if(calibSquareOn == 1){
		glTranslated(squareCenter[0],squareCenter[1],0.0);
		glBegin(GL_LINE_LOOP);
		glVertex3d(squareSize/2,squareSize/2,0.0);
		glVertex3d(-squareSize/2,squareSize/2,0.0);
		glVertex3d(-squareSize/2,-squareSize/2,0.0);
		glVertex3d(squareSize/2,-squareSize/2,0.0);
		glEnd();
		glTranslated(-squareCenter[0],-squareCenter[1],0.0);
	}


	// If we don't want the fixation point rotated, go ahead and draw it at
	// a fixed position in front of the camera.
	if (m_rotationType == 0) {
		glDisable(GL_STENCIL_TEST);
		glPointSize(FP_DOTSIZE);
		// Fixation point.
		if (m_starfield.drawFixationPoint == 1.0) {
			/***
			glColor3d(1.0*m_starfield.targLumMult[0], 0.0*m_starfield.targLumMult[0], 0.0*m_starfield.targLumMult[0]);
			glBegin(GL_POINTS);
			glVertex3d(m_starfield.fixationPointLocation[0] + m_Lateral,
			m_starfield.fixationPointLocation[1] - m_Heave,
			m_starfield.fixationPointLocation[2] - m_Surge);
			DrawTargetObject(DRAW_FP);
			glEnd();
			***/
			glPushMatrix();

#if USING_FISHEYE
				glTranslatef(m_starfield.fixationPointLocation[0] + m_Lateral,
				m_starfield.fixationPointLocation[1] - m_Heave,//+DY+adjusted_offset,
				m_starfield.fixationPointLocation[2] - m_Surge);
#else
			glTranslatef(m_starfield.fixationPointLocation[0] + m_Lateral,
				m_starfield.fixationPointLocation[1] - m_Heave,
				m_starfield.fixationPointLocation[2] - m_Surge);
#endif
			DrawTargetObject(DRAW_FP);
			glPopMatrix();
		}
		glEnable(GL_STENCIL_TEST);
	}

	// Target 1
	if (m_starfield.drawTarget1 == 1.0) {
		/***
		glColor3d(1.0*m_starfield.targLumMult[1], 0.0*m_starfield.targLumMult[1], 0.0*m_starfield.targLumMult[1]);
		glBegin(GL_POINTS);
		glVertex3d(m_starfield.targ1Location[0] + m_Lateral,
		m_starfield.targ1Location[1] - m_Heave,
		m_starfield.targ1Location[2] - m_Surge);
		DrawTargetObject(DRAW_TARG1);
		glEnd();
		***/
		glPushMatrix();
		glTranslatef(m_starfield.targ1Location[0] + m_Lateral,
			m_starfield.targ1Location[1] - m_Heave,
			m_starfield.targ1Location[2] - m_Surge);
		DrawTargetObject(DRAW_TARG1);
		glPopMatrix();
	}

	// Target 2
	if (m_starfield.drawTarget2 == 1.0) {
		/***
		glColor3d(1.0*m_starfield.targLumMult[2], 0.0*m_starfield.targLumMult[2], 0.0*m_starfield.targLumMult[2]);
		glBegin(GL_POINTS);
		glVertex3d(m_starfield.targ2Location[0] + m_Lateral,
		m_starfield.targ2Location[1] - m_Heave,
		m_starfield.targ2Location[2] - m_Surge);
		DrawTargetObject(DRAW_TARG2);
		glEnd();
		**/
		glPushMatrix();
		glTranslatef(m_starfield.targ2Location[0] + m_Lateral,
			m_starfield.targ2Location[1] - m_Heave,
			m_starfield.targ2Location[2] - m_Surge);
		DrawTargetObject(DRAW_TARG2);
		glPopMatrix();
	}

	//glPushMatrix();
	//glPopMatrix();

	// This is mainly used to calibrate the amplitude of sinusoid rotation movement
	// Drawing fixed max and min line of rotation amplitude.
	if(calibRotOn[0] == 1){
		glLineWidth(lineWidth);
		if (m_doRotation == true){	
			//glPushMatrix();
			glTranslated(m_centerX, m_centerY, m_centerZ);
			glRotated(minPredictedAngle, m_rotationVector.x, m_rotationVector.y, m_rotationVector.z);
			glTranslated(-m_centerX, -m_centerY, -m_centerZ);
			glColor3d(1.0, 1.0, 1.0);
			glBegin(GL_LINES);
			if(calibRotMotion == 0){ // horizontal line (pitch)
				glVertex3d(lineLength/2, 0.0, 0.0);
				glVertex3d(-lineLength/2, 0.0, 0.0);
			}
			else{// vertical line (yaw and roll)
				glVertex3d(0.0, lineLength/2, 0.0);
				glVertex3d(0.0, -lineLength/2, 0.0);
			}
			glEnd();
			//glPopMatrix();

			//glPushMatrix();
			glTranslated(m_centerX, m_centerY, m_centerZ);
			glRotated(maxPredictedAngle-minPredictedAngle, m_rotationVector.x, m_rotationVector.y, m_rotationVector.z);
			glTranslated(-m_centerX, -m_centerY, -m_centerZ);
			glBegin(GL_LINES);
			if(calibRotMotion == 0){ // horizontal line (pitch)
				glVertex3d(lineLength/2, 0.0, 0.0);
				glVertex3d(-lineLength/2, 0.0, 0.0);
			}
			else{// vertical line (yaw and roll)			
				glVertex3d(0.0, lineLength/2, 0.0);
				glVertex3d(0.0, -lineLength/2, 0.0);
			}
			glEnd();
			//glPopMatrix();

			glTranslated(m_centerX, m_centerY, m_centerZ);
			glRotated(-maxPredictedAngle, m_rotationVector.x, m_rotationVector.y, m_rotationVector.z);
			glTranslated(-m_centerX, -m_centerY, -m_centerZ);
		}
	}


	// If we're flagged to do so, rotate the the star field.
	if (m_doRotation == true) {
		glPushMatrix();
		glTranslated(m_centerX, m_centerY, m_centerZ);
		glRotated(m_rotationAngle, m_rotationVector.x, m_rotationVector.y, m_rotationVector.z);
		glTranslated(-m_centerX, -m_centerY, -m_centerZ);
	}


	// This is mainly used to calibrate the amplitude of sinusoid rotation movement
	// Drawing movement line
	if(calibRotOn[1] == 1){
		glLineWidth(lineWidth);
		if (m_doRotation == true){	
			glColor3d(1.0, 1.0, 1.0);
			glBegin(GL_LINES);
			if(calibRotMotion == 0){ // horizontal line (pitch)
				glVertex3d(lineLength/2, 0.0, 0.0);
				glVertex3d(-lineLength/2, 0.0, 0.0);
			}
			else{// vertical line (yaw and roll)
				glVertex3d(0.0, lineLength/2, 0.0);
				glVertex3d(0.0, -lineLength/2, 0.0);
			}
			glEnd();
		}
	}


	// Rotate the fixation point.  It will only be rotated if we're flagged to do a
	// rotation transformation.  Otherwise, it's just like the standard fixation point.
	if (m_rotationType == 1 || m_rotationType == 2) {
		glPointSize(FP_DOTSIZE);
		// Fixation point.
		if (m_starfield.drawFixationPoint == 1.0) {
			/***
			glColor3d(1.0, 0.0, 0.0);
			glBegin(GL_POINTS);
			glVertex3d(m_starfield.fixationPointLocation[0] + m_Lateral,
			m_starfield.fixationPointLocation[1] - m_Heave,
			m_starfield.fixationPointLocation[2] - m_Surge);
			DrawTargetObject(DRAW_FP);
			glEnd();
			***/
			glPushMatrix();
			glTranslatef(m_starfield.fixationPointLocation[0] + m_Lateral,
				m_starfield.fixationPointLocation[1] - m_Heave,
				m_starfield.fixationPointLocation[2] - m_Surge);
			DrawTargetObject(DRAW_FP);
			glPopMatrix();
		}
	}

	// If we're doing rotation, but we don't want to rotate the background,
	// pop off the rotated modelview matrix to get back to the normal one.
	if (m_rotationType == 2) {
		glPopMatrix();
	}

	// Draw the left starfield.
	glColor3d(m_starfield.starLeftColor[0] * m_starfield.luminance,		// Red
		m_starfield.starLeftColor[1] * m_starfield.luminance,		// Green
		m_starfield.starLeftColor[2] * m_starfield.luminance);	// Blue
	if (m_starfield.drawBackground == 1.0) {
		DrawStarField();
	}

	// Johnny add - 10/21/07
	if (m_doRotation == true) {
		glPopMatrix();
	}

	// Draw FP cross at Screen or at Edge of sphere
	if(FP_cross[0] == 1.0){ // FP cross On
		if(FP_cross[1] == 1.0) glTranslated(0.0, 0.0, m_frustum.camera2screenDist-hollowSphereRadius);
		if(FP_cross[4] == 1.0){ // add shadow cross
			glLineWidth(FP_cross[5]);
			glColor3d(0.0, 0.0, 0.0);
			glBegin(GL_LINES);
			// horizontal line
			glVertex3d(FP_cross[2]/2, 0.0, 0.0);
			glVertex3d(-FP_cross[2]/2, 0.0, 0.0);
			// vertical line
			glVertex3d(0.0, FP_cross[2]/2, 0.0);
			glVertex3d(0.0, -FP_cross[2]/2, 0.0);
			glEnd();
		}

		// draw the cross
		glLineWidth(FP_cross[3]);
		glColor3d(1.0, 0.0, 0.0);
		glBegin(GL_LINES);
		// horizontal line
		glVertex3d(FP_cross[2]/2, 0.0, 0.0);
		glVertex3d(-FP_cross[2]/2, 0.0, 0.0);
		// vertical line
		glVertex3d(0.0, FP_cross[2]/2, 0.0);
		glVertex3d(0.0, -FP_cross[2]/2, 0.0);
		glEnd();
		if(FP_cross[1] == 1.0) glTranslated(0.0, 0.0, -m_frustum.camera2screenDist+hollowSphereRadius);
	}

	//**** BACK RIGHT BUFFER. ****//
#if USE_STEREO
	glDrawBuffer(GL_BACK_RIGHT);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);		// Clears the current scene.
#else
	glColorMask(GL_FALSE, GL_TRUE, GL_FALSE, GL_FALSE);
#endif

#if USING_FISHEYE
    CalculateStereoFrustum(scrWidth,scrHeight, m_frustum.camera2screenDist,
		m_frustum.clipNear, m_frustum.clipFar, m_frustum.eyeSeparation / 2.0f,
		m_frustum.worldOffsetX, m_frustum.worldOffsetZ);
#else
	// Setup the projection matrix.
	CalculateStereoFrustum(m_frustum.screenWidth, m_frustum.screenHeight, m_frustum.camera2screenDist,
		m_frustum.clipNear, m_frustum.clipFar, m_frustum.eyeSeparation / 2.0f,
		m_frustum.worldOffsetX, m_frustum.worldOffsetZ);
#endif
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

#if USING_FISHEYE
	// Setup the camera.
	gluLookAt(m_frustum.eyeSeparation/2.0f+m_Lateral, 0.0f-m_Heave+DY+adjusted_offset, m_frustum.camera2screenDist-m_Surge,		// Camera origin
		m_frustum.eyeSeparation/2.0f+m_Lateral, 0.0f-m_Heave+DY+adjusted_offset, m_frustum.camera2screenDist-m_Surge-1.0f,	// Camera direction
		0.0, 1.0, 0.0);																					// Which way is up
#else
	// Setup the camera.
	gluLookAt(m_frustum.eyeSeparation/2.0f+m_Lateral, 0.0f-m_Heave, m_frustum.camera2screenDist-m_Surge,		// Camera origin
		m_frustum.eyeSeparation/2.0f+m_Lateral, 0.0f-m_Heave, m_frustum.camera2screenDist-m_Surge-1.0f,	// Camera direction
		0.0, 1.0, 0.0);																					// Which way is up
#endif

#if USE_STEREO
	// If we are using the cutout, we need to setup the stencil buffer.
	if (m_starfield.useCutout == true) {
		// Turn off polygon smoothing otherwise we get weird lines in the
		// triangle fan.
		glDisable(GL_POLYGON_SMOOTH);

		// Use 0 for clear stencil, enable stencil test
		glClearStencil(0);
		glEnable(GL_STENCIL_TEST);

		// All drawing commands fail the stencil test, and are not
		// drawn, but increment the value in the stencil buffer.
		glStencilFunc(GL_NEVER, 0x0, 0x0);
		glStencilOp(GL_INCR, GL_INCR, GL_INCR);

		// Draw a circle.
		glColor3d(1.0, 1.0, 1.0);
		glBegin(GL_TRIANGLE_FAN);
		//glVertex2d(0.0, 0.0);
		for(double dAngle = 0; dAngle <= 360.0; dAngle += 15.0) {
			glVertex3d(m_starfield.cutoutRadius * cos(dAngle*DEG2RAD) + m_starfield.fixationPointLocation[0] + m_Lateral,
				m_starfield.cutoutRadius * sin(dAngle*DEG2RAD) + m_starfield.fixationPointLocation[1] - m_Heave,
				m_starfield.fixationPointLocation[2] - m_Surge);
		}
		glEnd();

		// Now, allow drawing, except where the stencil pattern is 0x1
		// and do not make any further changes to the stencil buffer
		glStencilFunc(GL_NOTEQUAL, 0x1, 0x1);
		glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

		// Turn smoothing back on to draw the star field.
		glEnable(GL_POLYGON_SMOOTH);
	}
	else {
		glDisable(GL_STENCIL_TEST);
	}
#endif

	// Create the center dot.  This is not the same as the fixation target.
	// This is mainly used to calibrate movement.
	glTranslated(calibTranOn[2],calibTranOn[3],calibTranOn[4]);
	glLineWidth(lineWidth);
	glColor3d(0.0, 1.0, 0.0);
	glBegin(GL_LINES);
	// horizontal line
	if(calibTranOn[0]== 1){
		glVertex3d(lineLength/2, 0.0, 0.0);
		glVertex3d(-lineLength/2, 0.0, 0.0);
	}
	// vertical line
	if(calibTranOn[1]== 1){
		glVertex3d(0.0, lineLength/2, 0.0);
		glVertex3d(0.0, -lineLength/2, 0.0);
	}
	glEnd();
	glTranslated(-calibTranOn[2],-calibTranOn[3],-calibTranOn[4]);

	// If we don't want the fixation point rotated, go ahead and draw it at
	// a fixed position in front of the camera.
	if (m_rotationType == 0) {
		glDisable(GL_STENCIL_TEST);
		glPointSize(FP_DOTSIZE);
		// Fixation point.
		if (m_starfield.drawFixationPoint == 1.0) {

			/***
			glColor3d(0.0*m_starfield.targLumMult[0], 1.0*m_starfield.targLumMult[0], 0.0*m_starfield.targLumMult[0]);
			glBegin(GL_POINTS);
			glVertex3d(m_starfield.fixationPointLocation[0] + m_Lateral,
			m_starfield.fixationPointLocation[1] - m_Heave,
			m_starfield.fixationPointLocation[2] - m_Surge);
			DrawTargetObject(DRAW_FP);
			glEnd();
			***/
			glPushMatrix();

#if USING_FISHEYE
				glTranslatef(m_starfield.fixationPointLocation[0] + m_Lateral,
				m_starfield.fixationPointLocation[1] - m_Heave,//+DY+adjusted_offset,
				m_starfield.fixationPointLocation[2] - m_Surge);
#else
			glTranslatef(m_starfield.fixationPointLocation[0] + m_Lateral,
				m_starfield.fixationPointLocation[1] - m_Heave,
				m_starfield.fixationPointLocation[2] - m_Surge);
#endif
			DrawTargetObject(DRAW_FP);
			glPopMatrix();
		}
		glEnable(GL_STENCIL_TEST);
	}

	// Target 1
	if (m_starfield.drawTarget1 == 1.0) {
		/***
		glColor3d(0.0*m_starfield.targLumMult[1], 1.0*m_starfield.targLumMult[1], 0.0*m_starfield.targLumMult[1]);
		glBegin(GL_POINTS);
		glVertex3d(m_starfield.targ1Location[0] + m_Lateral,
		m_starfield.targ1Location[1] - m_Heave,
		m_starfield.targ1Location[2] - m_Surge);
		DrawTargetObject(DRAW_TARG1);
		glEnd();
		***/

		glPushMatrix();
		glTranslatef(m_starfield.targ1Location[0] + m_Lateral,
			m_starfield.targ1Location[1] - m_Heave,
			m_starfield.targ1Location[2] - m_Surge);
		DrawTargetObject(DRAW_TARG1);
		glPopMatrix();
	}

	// Target 2
	if (m_starfield.drawTarget2 == 1.0) {
		/***
		glColor3d(0.0*m_starfield.targLumMult[2], 1.0*m_starfield.targLumMult[2], 0.0*m_starfield.targLumMult[2]);
		glBegin(GL_POINTS);
		glVertex3d(m_starfield.targ2Location[0] + m_Lateral,
		m_starfield.targ2Location[1] - m_Heave,
		m_starfield.targ2Location[2] - m_Surge);
		DrawTargetObject(DRAW_TARG2);
		glEnd();
		***/

		glPushMatrix();
		glTranslatef(m_starfield.targ2Location[0] + m_Lateral,
			m_starfield.targ2Location[1] - m_Heave,
			m_starfield.targ2Location[2] - m_Surge);
		DrawTargetObject(DRAW_TARG2);
		glPopMatrix();

	}

	// If we're flagged to do so, rotate the the star field.
	if (m_doRotation == true) {
		glPushMatrix();
		glTranslated(m_centerX, m_centerY, m_centerZ);
		glRotated(m_rotationAngle, m_rotationVector.x, m_rotationVector.y, m_rotationVector.z);
		glTranslated(-m_centerX, -m_centerY, -m_centerZ);
	}

	// Rotate the fixation point.  It will only be rotated if we're flagged to do a
	// rotation transformation.  Otherwise, it's just like the standard fixation point.
	if (m_rotationType == 1 || m_rotationType == 2) {
		glPointSize(FP_DOTSIZE);
		// Fixation point.
		if (m_starfield.drawFixationPoint == 1.0) {
			/***
			glColor3d(0.0, 1.0, 0.0);
			glBegin(GL_POINTS);
			glVertex3d(m_starfield.fixationPointLocation[0] + m_Lateral,
			m_starfield.fixationPointLocation[1] - m_Heave,
			m_starfield.fixationPointLocation[2] - m_Surge);
			DrawTargetObject(DRAW_FP);
			glEnd();
			***/

			glPushMatrix();
			glTranslatef(m_starfield.fixationPointLocation[0] + m_Lateral,
				m_starfield.fixationPointLocation[1] - m_Heave,
				m_starfield.fixationPointLocation[2] - m_Surge);
			DrawTargetObject(DRAW_FP);
			glPopMatrix();
		}
	}

	// If we're doing rotation, but we don't want to rotate the background,
	// pop off the rotated modelview matrix to get back to the normal one.
	if (m_rotationType == 2) {
		glPopMatrix();
	}

	// Draw the right starfield.
	glColor3d(m_starfield.starRightColor[0] * m_starfield.luminance,		// Red
		m_starfield.starRightColor[1] * m_starfield.luminance,		// Green
		m_starfield.starRightColor[2] * m_starfield.luminance);		// Blue
	if (m_starfield.drawBackground == 1.0) {
		DrawStarField();
	}

	// Johnny add - 10/21/07
	if (m_doRotation == true) {
		glPopMatrix();
	}

	// Draw FP cross at Screen or at Edge of sphere
	if(FP_cross[0] == 1.0){ // FP cross On
		if(FP_cross[1] == 1.0) glTranslated(0.0, 0.0, m_frustum.camera2screenDist-hollowSphereRadius);
		if(FP_cross[4] == 1.0){ // add shadow cross
			glLineWidth(FP_cross[5]);
			glColor3d(0.0, 0.0, 0.0);
			glBegin(GL_LINES);
			// horizontal line
			glVertex3d(FP_cross[2]/2, 0.0, 0.0);
			glVertex3d(-FP_cross[2]/2, 0.0, 0.0);
			// vertical line
			glVertex3d(0.0, FP_cross[2]/2, 0.0);
			glVertex3d(0.0, -FP_cross[2]/2, 0.0);
			glEnd();
		}

		// draw the cross
		glLineWidth(FP_cross[3]);
		glColor3d(0.0, 1.0, 0.0);
		glBegin(GL_LINES);
		// horizontal line
		glVertex3d(FP_cross[2]/2, 0.0, 0.0);
		glVertex3d(-FP_cross[2]/2, 0.0, 0.0);
		// vertical line
		glVertex3d(0.0, FP_cross[2]/2, 0.0);
		glVertex3d(0.0, -FP_cross[2]/2, 0.0);
		glEnd();
		if(FP_cross[1] == 1.0) glTranslated(0.0, 0.0, -m_frustum.camera2screenDist+hollowSphereRadius);
	}

	//glFlush();
	glFinish();
}


void GLPanel::DoRotation(bool val)
{
	m_doRotation = val;
}


void GLPanel::SetRotationVector(nm3DDatum rotationVector)
{
	m_rotationVector = rotationVector;
}


void GLPanel::SetRotationAngle(double angle)
{
	m_rotationAngle = angle;
}


void GLPanel::SetRotationCenter(double x, double y, double z)
{
	m_centerX = x;
	m_centerY = y;
	m_centerZ = z;
}


GLvoid GLPanel::DrawStarField()
{	
	/* Chris original code
	int i;

	// Don't try to mess with an unallocated array.
	if (m_starArray == NULL) {
	return;
	}

	for (i = 0; i < m_starfield.totalStars; i++) {
	glBegin(GL_TRIANGLES);
	glVertex3d(m_starArray[i].x[0], m_starArray[i].y[0], m_starArray[i].z[0]);
	glVertex3d(m_starArray[i].x[1], m_starArray[i].y[1], m_starArray[i].z[1]);
	glVertex3d(m_starArray[i].x[2], m_starArray[i].y[2], m_starArray[i].z[2]);
	glEnd();
	}
	*/

	if (g_pList.GetVectorData("DRAW_MODE").at(0) == 1.0){ // triangles' cube
		int j = m_starfield.totalStars*3*3;
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, 0, starFieldVertex3D);
		glDrawArrays(GL_TRIANGLES,0,j/3);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	else if (g_pList.GetVectorData("DRAW_MODE").at(0) == 0.0){ // Dots' sphere
		glPointSize(g_pList.GetVectorData("DOTS_SIZE").at(0));
		int j = m_starfield.totalStars*3;
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, 0, starFieldVertex3D);
		glDrawArrays(GL_POINTS,0,j/3);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	else if (g_pList.GetVectorData("DRAW_MODE").at(0) == 2.0){ // Dots' cube
		glPointSize(g_pList.GetVectorData("DOTS_SIZE").at(0));
		int j = m_starfield.totalStars*3;
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, 0, starFieldVertex3D);
		glDrawArrays(GL_POINTS,0,j/3);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	else if (g_pList.GetVectorData("DRAW_MODE").at(0) == 3.0){ // Dots' hollow shpere

		if(!glIsList(m_starFieldCallList)) SetupCallList();
		if(redrawCallList){
			glDeleteLists(m_starFieldCallList,1);
			SetupCallList();
			redrawCallList = false;
		}
		glCallList(m_starFieldCallList);
	}
}

GLvoid GLPanel::SetupCallList()
{
	int pixelWidth = 1280;	// viewport pixel width
	if(RIG_ROOM == MOOG_B54) pixelWidth = 782;	
	else if(RIG_ROOM == MOOG_217) pixelWidth = 1010;
	else if(RIG_ROOM == MOOG_B55) pixelWidth = 1400;
	double radius = g_pList.GetVectorData("DOTS_SPHERE_PARA").at(0);

	int totalDots = 0;
	double starRadius = 5.0; //cm
	double starInc = 360.0/20.0;

	glNewList(m_starFieldCallList, GL_COMPILE);
	glPushMatrix();
	// translate everything to the center (eye location)
	glTranslatef(m_frustum.worldOffsetX,m_frustum.worldOffsetZ,m_frustum.camera2screenDist);
	for(int i = 0; i<(int)dotSizeDistribute.size(); i++){
		starRadius = ((double)(i+1)/(double)pixelWidth*m_frustum.screenWidth)/2.0;
		for(int j = 0; j < (int)dotSizeDistribute.at(i); j++){
			glPushMatrix();
#if USING_FISHEYE
			double x0=radius*cos(aziVector.at(totalDots+j))*cos(eleVector.at(totalDots+j));
            double y0=radius*sin(aziVector.at(totalDots+j))*cos(eleVector.at(totalDots+j));
			double z0=radius*sin(eleVector.at(totalDots+j));
            glBegin(GL_TRIANGLE_FAN); // draw circle in starRadius at origin on XY plan
			glVertex3d(0.0, 0.0, 0.0);
            for(double dAngle = 10; dAngle <= 370.0; dAngle += starInc) {
				double x=x0+starRadius * cos(dAngle*DEG2RAD);
				double y=y0+starRadius * sin(dAngle*DEG2RAD);
				double z=z0;
			}
			glEnd();
#else
			glRotatef(aziVector.at(totalDots+j)*RAD2DEG,0,1,0); // rotate at y-axis
			glRotatef(eleVector.at(totalDots+j)*RAD2DEG,0,0,1); // rotate at z-axis
			glTranslatef(radius,0,0);
			glRotatef(90,0,1,0);
			glBegin(GL_TRIANGLE_FAN); // draw circle in starRadius at origin on XY plan
			glVertex3d(0.0, 0.0, 0.0);
			for(double dAngle = 10; dAngle <= 370.0; dAngle += starInc) {
				glVertex3d(starRadius * cos(dAngle*DEG2RAD) + 0.0, //x
					starRadius * sin(dAngle*DEG2RAD) + 0.0, //y
					0.0); //z
			}
			glEnd();
#endif
			glPopMatrix();
		}
		totalDots+=dotSizeDistribute.at(i);
	}

	glPopMatrix();
	glEndList();

}

GLvoid GLPanel::ModifyStarField()
{
	int i;
	double baseX, baseY, baseZ, prob,
		sd0, sd1, sd2,
		ts0, ts1;

	// Grab the starfield dimensions and triangle size.  Pulling this stuff out
	// of the vectors now produces a 20 fold increase in speed in the following
	// loop.
	sd0 = m_starfield.dimensions[0];
	sd1 = m_starfield.dimensions[1];
	sd2 = m_starfield.dimensions[2];
	ts0 = m_starfield.triangle_size[0];
	ts1 = m_starfield.triangle_size[1];

	// don't draw any dots inside the m_frustum.clipNear
	// put all dots between two spheres
	double min = m_frustum.clipNear*m_frustum.clipNear*
		(1+(m_frustum.screenWidth*m_frustum.screenWidth+m_frustum.screenHeight*m_frustum.screenHeight)/
		m_frustum.camera2screenDist/m_frustum.camera2screenDist/4);
	double minDist = sd0;
	if(minDist > sd1) minDist = sd1;
	if(minDist > sd2) minDist = sd2;
	double max = minDist*minDist/4;

	int j = 0;
	for (i = 0; i < m_starfield.totalStars; i++) {
		// If a star is in our probability range, we'll modify it.
		prob = (double)rand()/(double)RAND_MAX*100.0;
		//double prob = 100.0;

		// If the coherence factor is higher than a random number chosen between
		// 0 and 100, then we don't do anything to the star.  This means that
		// (100-coherence)% of the total stars will change.
		if (m_starfield.probability < prob) {
			// Find a random point to base our triangle around.
			baseX = (double)rand()/(double)RAND_MAX*sd0 - sd0/2.0;
			baseY = (double)rand()/(double)RAND_MAX*sd1 - sd1/2.0;
			baseZ = (double)rand()/(double)RAND_MAX*sd2 - sd2/2.0;

			// Vertex 1
			m_starArray[i].x[0] = baseX - ts0/2.0;
			m_starArray[i].y[0] = baseY - ts1/2.0;
			m_starArray[i].z[0] = baseZ;

			// Vertex 2
			m_starArray[i].x[1] = baseX;
			m_starArray[i].y[1] = baseY + ts1/2.0;
			m_starArray[i].z[1] = baseZ;

			// Vertex 3
			m_starArray[i].x[2] = baseX + ts0/2.0;
			m_starArray[i].y[2] = baseY - ts1/2.0;
			m_starArray[i].z[2] = baseZ;
		}

		// prepare glarray
		if (m_starfield.drawMode == 1.0){ // triangles
			starFieldVertex3D[j++] = m_starArray[i].x[0]; starFieldVertex3D[j++] = m_starArray[i].y[0]; starFieldVertex3D[j++] = m_starArray[i].z[0];
			starFieldVertex3D[j++] = m_starArray[i].x[1]; starFieldVertex3D[j++] = m_starArray[i].y[1]; starFieldVertex3D[j++] = m_starArray[i].z[1];
			starFieldVertex3D[j++] = m_starArray[i].x[2]; starFieldVertex3D[j++] = m_starArray[i].y[2]; starFieldVertex3D[j++] = m_starArray[i].z[2];
		}
		else if (m_starfield.drawMode == 0.0){ // Dots
			// borrow star field big array
			// don't draw any dots inside the m_frustum.clipNear
			// put all dots between two spheres and shift to the eye center
			if (m_starfield.probability < prob){ // using new location
				if((baseX*baseX + baseY*baseY + baseZ*baseZ > min) && (baseX*baseX + baseY*baseY + baseZ*baseZ < max)){
					starFieldVertex3D[j++] = baseX+m_frustum.worldOffsetX; // Horizontal
					starFieldVertex3D[j++] = baseY+m_frustum.worldOffsetZ; // Vertical
					starFieldVertex3D[j++] = baseZ+m_frustum.camera2screenDist;
				}
				else i--;
			}
			else j = j+3; // keep old location
		}

	}
}


GLvoid GLPanel::GenerateStarField()
{
	int i;
	double baseX, baseY, baseZ,
		sd0, sd1, sd2,
		ts0, ts1;
	double radius, density, sigma, mean, azimuth, elevation;
	//vector <double> aziVector, eleVector;

	// Delete the old Star array if needed.
	if (m_starArray != NULL) {
		delete [] m_starArray;
	}

	// Seed the random number generator.
	srand((unsigned)time(NULL));

	// Determine the total number of stars needed to create an average density determined
	// from the StarField structure.
	if (m_starfield.drawMode == 3.0){ // Dots' hollow sphere
		radius = g_pList.GetVectorData("DOTS_SPHERE_PARA").at(0);
		density = g_pList.GetVectorData("DOTS_SPHERE_PARA").at(1);
		m_starfield.totalStars = 4*PI*radius*radius*density;
	}
	else
		m_starfield.totalStars = (int)(m_starfield.dimensions[0] * m_starfield.dimensions[1] *
		m_starfield.dimensions[2] * m_starfield.density);

	// Allocate the Star array.
	m_starArray = new Star[m_starfield.totalStars];

	// Grab the starfield dimensions and triangle size.  Pulling this stuff out
	// of the vectors now produces a 20 fold increase in speed in the following
	// loop.
	sd0 = m_starfield.dimensions[0];
	sd1 = m_starfield.dimensions[1];
	sd2 = m_starfield.dimensions[2];
	ts0 = m_starfield.triangle_size[0];
	ts1 = m_starfield.triangle_size[1];

	// using glarray to draw star field
	if(starFieldVertex3D != NULL){
		delete [] starFieldVertex3D;
	}
	starFieldVertex3D = new GLfloat[m_starfield.totalStars*3*3];

	// don't draw any dots inside the m_frustum.clipNear
	// put all dots between two spheres
	double min = m_frustum.clipNear*m_frustum.clipNear*
		(1+(m_frustum.screenWidth*m_frustum.screenWidth+m_frustum.screenHeight*m_frustum.screenHeight)/
		m_frustum.camera2screenDist/m_frustum.camera2screenDist/4);
	double minDist = sd0;
	if(minDist > sd1) minDist = sd1;
	if(minDist > sd2) minDist = sd2;
	double max = minDist*minDist/4;

	aziVector.clear(); eleVector.clear();
	int j = 0;
	for (i = 0; i < m_starfield.totalStars; i++) {

		if (m_starfield.drawMode == 3.0){ // drawing Dots' hollow sphere -- monkey eye at center
			int tmp = j%2;
			// Find a random point on hollow sphere.
			azimuth = ((double)rand()/(double)RAND_MAX)*2*PI;			//0 <= azimth <= 360
			elevation = asin(((double)rand()/(double)RAND_MAX)*2-1);	//-90(-1) <= elevation <= 90(1); The Lambert Projection
			//starFieldVertex3D[j++] = radius*cos(elevation)*cos(azimth) + m_frustum.worldOffsetX; // Horizontal (X)
			//starFieldVertex3D[j++] = radius*sin(elevation) + m_frustum.worldOffsetZ; // Vertical (Y)
			//starFieldVertex3D[j++] = radius*cos(elevation)*sin(azimth) + m_frustum.camera2screenDist; // (Z)
			aziVector.push_back(azimuth);
			eleVector.push_back(elevation);
		}
		else{
			// Find a random point to base our triangle around.
			baseX = (double)rand()/(double)RAND_MAX*sd0 - sd0/2.0;
			baseY = (double)rand()/(double)RAND_MAX*sd1 - sd1/2.0;
			baseZ = (double)rand()/(double)RAND_MAX*sd2 - sd2/2.0;

			/***
			// Vertex 1
			m_starArray[i].x[0] = baseX - ts0/2.0;
			m_starArray[i].y[0] = baseY - ts1/2.0;
			m_starArray[i].z[0] = baseZ;

			// Vertex 2
			m_starArray[i].x[1] = baseX;
			m_starArray[i].y[1] = baseY + ts1/2.0;
			m_starArray[i].z[1] = baseZ;

			// Vertex 3
			m_starArray[i].x[2] = baseX + ts0/2.0;
			m_starArray[i].y[2] = baseY - ts1/2.0;
			m_starArray[i].z[2] = baseZ;
			***/

			m_starArray[i].x[0] = baseX - ts0/2.0;
			m_starArray[i].y[0] = baseY + ts1/2.0;
			m_starArray[i].z[0] = baseZ;

			// Vertex 2
			m_starArray[i].x[1] = baseX;
			m_starArray[i].y[1] = baseY - ts1/2.0;
			m_starArray[i].z[1] = baseZ;

			// Vertex 3
			m_starArray[i].x[2] = baseX + ts0/2.0;
			m_starArray[i].y[2] = baseY + ts1/2.0;
			m_starArray[i].z[2] = baseZ;
		}

		if (m_starfield.drawMode == 1.0){ // Triangles' cube
			// prepare glarray
#if USING_FISHEYE
			POS3D p3d[3];
			for(int pp=0;pp<3;pp++){
				p3d[pp].x=m_starArray[i].x[pp];
				p3d[pp].y=m_starArray[i].y[pp];
				p3d[pp].z=m_starArray[i].z[pp];
				p3d[pp].x /= (sd0/2.0);
				p3d[pp].y /= (sd1/2.0);
				p3d[pp].z /= (sd2/2.0);
			}
			double u[3],v[3];
            if(getWarppedForFishEye(p3d[0],u[0],v[0])&&getWarppedForFishEye(p3d[1],u[1],v[1])&&getWarppedForFishEye(p3d[2],u[2],v[2])){
				for(int pp=0;pp<3;pp++){
					starFieldVertex3D[j++] = u[pp];
                    starFieldVertex3D[j++] = v[pp];
					starFieldVertex3D[j++] = p3d[pp].z*(sd2/2.0);//0.0;
				}
			}
#else
			starFieldVertex3D[j++] = m_starArray[i].x[0]; starFieldVertex3D[j++] = m_starArray[i].y[0]; starFieldVertex3D[j++] = m_starArray[i].z[0];
			starFieldVertex3D[j++] = m_starArray[i].x[1]; starFieldVertex3D[j++] = m_starArray[i].y[1]; starFieldVertex3D[j++] = m_starArray[i].z[1];
			starFieldVertex3D[j++] = m_starArray[i].x[2]; starFieldVertex3D[j++] = m_starArray[i].y[2]; starFieldVertex3D[j++] = m_starArray[i].z[2];
#endif
		}
		else if (m_starfield.drawMode == 0.0){ // Dots' sphere
			// borrow star field big array
			// don't draw any dots inside the m_frustum.clipNear
			// put all dots between two spheres and shift to the eye center
			if((baseX*baseX + baseY*baseY + baseZ*baseZ > min) && (baseX*baseX + baseY*baseY + baseZ*baseZ < max)){
				starFieldVertex3D[j++] = baseX+m_frustum.worldOffsetX; // Horizontal
				starFieldVertex3D[j++] = baseY+m_frustum.worldOffsetZ;//+GL_OFFSET; // Vertical
				starFieldVertex3D[j++] = baseZ+m_frustum.camera2screenDist;
			}
			else i--;
		}
		else if (m_starfield.drawMode == 2.0){ // Dots' cube
			starFieldVertex3D[j++] = baseX; 
			starFieldVertex3D[j++] = baseY;
			starFieldVertex3D[j++] = baseZ;
		}
	}

	// setup the vector to record how many dots at different size(glPointSize()).
	if (m_starfield.drawMode == 3.0){ // drawing Dots' hollow sphere
		double gaussPercentage = 0.0;
		int totalDots = 0;
		int numOfDots = 0;
		int size = 1;

		mean = g_pList.GetVectorData("DOTS_SPHERE_PARA").at(2);
		sigma = g_pList.GetVectorData("DOTS_SPHERE_PARA").at(3);
		// Create a conversion factor to convert from degrees
		// into centimeters for OpenGL.
		mean = 2*tan(mean/2/180*PI)*(m_frustum.camera2screenDist);
		sigma = 2*tan(sigma/2/180*PI)*(m_frustum.camera2screenDist);
		int pixelWidth = 1280;	// viewport pixel width
		if(RIG_ROOM == MOOG_B54) pixelWidth = 782;	
		else if(RIG_ROOM == MOOG_217) pixelWidth = 1010;
		else if(RIG_ROOM == MOOG_B55) pixelWidth = 1400;
		// Change cm to pixel.
		mean = mean/m_frustum.screenWidth*pixelWidth;
		sigma = sigma/m_frustum.screenWidth*pixelWidth;

		dotSizeDistribute.clear();
		while(totalDots < m_starfield.totalStars){
			// use gaussian distribution find the percentage of total dots at certain size
			gaussPercentage = (1/sigma/sqrt(2*PI))*exp(-0.5*pow(((size-mean)/sigma),2.0));
			numOfDots = (int)(gaussPercentage*m_starfield.totalStars);
			// if we still have some dots left after we try different sizes,
			// then set the rest of dots at mean size
			if (size > mean && numOfDots < 1){ 
				dotSizeDistribute.at((int)(mean-0.5)) = dotSizeDistribute.at((int)(mean-0.5)) + m_starfield.totalStars - totalDots;
				//int tmp = dotSizeDistribute.at((int)(mean-0.5)); // for checking only
				break;
			}

			totalDots += numOfDots;
			if (totalDots >= m_starfield.totalStars){
				dotSizeDistribute.push_back(m_starfield.totalStars - totalDots + numOfDots);
				break;
			}
			else dotSizeDistribute.push_back(numOfDots);

			size++;
		}

		SetupCallList();
	}
}


GLvoid GLPanel::SetFrustum(Frustum frustum)
{
	m_frustum = frustum;
}


GLvoid GLPanel::SetStarField(StarField starfield)
{
	m_starfield = starfield;

	// Regenerate the starfield based on new data.
	GenerateStarField();
}


GLvoid GLPanel::CalculateStereoFrustum(GLfloat screenWidth, GLfloat screenHeight, GLfloat camera2screenDist,
									   GLfloat clipNear, GLfloat clipFar, GLfloat eyeSeparation,
									   GLfloat centerOffsetX, GLfloat centerOffsetY)
{
	GLfloat left, right, top, bottom;

	// Go to projection mode.
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	// We use similar triangles to solve for the left, right, top, and bottom of the clipping
	// plane.
	top = (clipNear / camera2screenDist) * (screenHeight / 2.0f - centerOffsetY);
	bottom = (clipNear / camera2screenDist) * (-screenHeight / 2.0f - centerOffsetY);
	right = (clipNear / camera2screenDist) * (screenWidth / 2.0f - eyeSeparation - centerOffsetX);
	left = (clipNear / camera2screenDist) * (-screenWidth / 2.0f - eyeSeparation - centerOffsetX);

	glFrustum(left, right, bottom, top, clipNear, clipFar);	
}


void GLPanel::OnSize(wxSizeEvent &event)
{
	// this is also necessary to update the context on some platforms
	wxGLCanvas::OnSize(event);

	// Set GL viewport (not called by wxGLCanvas::OnSize on all platforms...).
	int w, h;
	GetClientSize(&w, &h);
	if (GetContext())
	{
		SetCurrent();
		glViewport(0, 0, (GLint) w, (GLint) h);
	}
}


void GLPanel::OnPaint(wxPaintEvent &event)
{
	wxPaintDC dc(this);

	// Make sure that we render the OpenGL stuff.
	SetCurrent();
	Render();
	SwapBuffers();
}


void GLPanel::InitGL(void)
{
	glClearColor(0.0, 0.0, 0.0, 0.0);					// Set background to black.
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glMatrixMode(GL_MODELVIEW);

#if USE_STEREO
	// Enable depth testing.
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
#endif

#if USE_ANTIALIASING
	// Enable Antialiasing
	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	glEnable(GL_POLYGON_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
#endif

	//#if WEIRD_MONITOR
	if(RIG_ROOM == MOOG_B54)
	{
		int w = 782, h = 897;  // New viewport settings.  JWN
		glViewport((SCREEN_WIDTH-w)/2+10, (SCREEN_HEIGHT-h)/2-7, w, h);
	}
	else if(RIG_ROOM == MOOG_217)
	{
		int w = 1010, h = 1024;  // New viewport settings.  JWN
		glViewport((SCREEN_WIDTH-w)/2+90, (SCREEN_HEIGHT-h)/2, w, h-7);
	}
#if !USING_FULL_SCREEN
	else if(RIG_ROOM == MINI_MOOG)
	{
		int w = 778, h = 768;  // New viewport settings.  JWN
		glViewport((SCREEN_WIDTH-w)/2-9, (SCREEN_HEIGHT-h)/2, w, h);
	}
#endif
	//#endif

#if JOHNNY_WORKSTATION
	glViewport(0,0,1280,1024);
#endif
#if !DUAL_MONITORS
	glViewport(0,0,800, 870);
#endif
}

void GLPanel::RotationType(int val)
{
	m_rotationType = val;
}

GLvoid GLPanel::DrawTargetObject(int targObj)
{
	glColor3d(	m_starfield.targRlum.at(targObj)*m_starfield.targLumMult.at(targObj), 
		m_starfield.targGlum.at(targObj)*m_starfield.targLumMult.at(targObj), 
		m_starfield.targBlum.at(targObj)*m_starfield.targLumMult.at(targObj));
	//// We need turn off GL_POLYGON_SMOOTH, because either polygon or rectangle
	//// will have line inside with background color.
	glDisable(GL_POLYGON_SMOOTH);
	if (m_starfield.targShape.at(targObj) == SHAPE_ELLIPSE)
	{
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, 0, m_starfield.targVertex[targObj]);
		glDrawArrays(GL_POLYGON,0,DRAW_TARG_SLICES);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	else if(m_starfield.targShape.at(targObj) == SHAPE_RECTANGLE)
	{
		glRectf(-m_starfield.targXsize.at(targObj)/2.0, -m_starfield.targYsize.at(targObj)/2.0,
			m_starfield.targXsize.at(targObj)/2.0, m_starfield.targYsize.at(targObj)/2.0);
	}
	else if(m_starfield.targShape.at(targObj) == SHAPE_PLUS)
	{
		glLineWidth(FP_cross[3]);
		glBegin(GL_LINES);
		glVertex3d(m_starfield.targXsize.at(targObj)/2.0, 0.0, 0.0);
		glVertex3d(-m_starfield.targXsize.at(targObj)/2.0, 0.0, 0.0);
		glVertex3d(0.0, m_starfield.targYsize.at(targObj)/2.0, 0.0);
		glVertex3d(0.0, -m_starfield.targYsize.at(targObj)/2.0, 0.0);
		glEnd();
	}
	else if(m_starfield.targShape.at(targObj) == SHAPE_CROSS)
	{
		glLineWidth(FP_cross[3]);
		glBegin(GL_LINES);
		glVertex3d(m_starfield.targXsize.at(targObj)/2.0, m_starfield.targYsize.at(targObj)/2.0, 0.0);
		glVertex3d(-m_starfield.targXsize.at(targObj)/2.0, -m_starfield.targYsize.at(targObj)/2.0, 0.0);
		glVertex3d(m_starfield.targXsize.at(targObj)/2.0, -m_starfield.targYsize.at(targObj)/2.0, 0.0);
		glVertex3d(-m_starfield.targXsize.at(targObj)/2.0, m_starfield.targYsize.at(targObj)/2.0, 0.0);
		glEnd();
	}
	glEnable(GL_POLYGON_SMOOTH);
}

#if USING_FISHEYE

bool getWarppedForFishEye(POS3D p, double &u, double &v)
{
	double r= p.x*p.x+p.y*p.y;
	if(r>1.0){
		return false;
	}
	p.z = sqrt(1 - r);
	double theta=atan2(p.z,p.x);
	double phi=atan2(sqrt(p.x*p.x+p.z*p.z),p.y);
	r=phi / PID2;
	u=(1 + r * cos(theta)) / 2-0.5;
	v=(1 + r * sin(theta)) / 2-0.5;

	u*=86.39379;
	v*=34.55752;

	return true;
}

#endif