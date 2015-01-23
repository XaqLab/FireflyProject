#pragma once


#define USE_ANTIALIASING 1		// 1 = use anti-aliasing, 0 = don't

#define FP_DOTSIZE 5			// Fixation point size

#define DRAW_FP 0
#define DRAW_TARG1 1
#define DRAW_TARG2 2
#define DRAW_TARG_SLICES 100
#define SHAPE_ELLIPSE	0
#define SHAPE_RECTANGLE	1
#define SHAPE_PLUS		2
#define SHAPE_CROSS		3

using namespace std;

class GLPanel;

// Defines a stereo glFrustum.
typedef struct FRUSTUM_STRUCT
{
	GLfloat screenWidth,		// Width of the screen.
		    screenHeight,		// Height of the screen.
			camera2screenDist,	// Distance from the camera to the screen.
			clipNear,			// Distance from camera to near clipping plane.
			clipFar,			// Distance from camera to far clipping plane.
			eyeSeparation,		// Interocular distance
			worldOffsetX,		// Shifts the entire world horizontally.
			worldOffsetZ;		// Shifts the entire world vertically.
} Frustum;

// Defines a 3D field of stars.
typedef struct STARFIELD_STRUCT
{
	vector<double> dimensions,				// Width, height, depth dimensions of starfield.
				   triangle_size,			// Base width, height for each star.
				   fixationPointLocation,	// (x,y,z) origin of fixation point.
				   targ1Location,			// (x,y,z) origin of target 1.
				   targ2Location,			// (x,y,z) origin of target 2.
				   targLumMult,					// luminous of fixationPoint, targ1 and targ2 
				   starLeftColor,			// (r,g,b) value of left eye starfield.
				   starRightColor,			// (r,g,b) value of right eye starfield.
				   targXsize,				// X-dimension(cm) of FP & targets // following add by Johnny - 11/6/08
				   targYsize,				// Y-dimension(cm) of FP & targets
				   targShape,				// shape of FP & targets: ELLIPSE or RECTANGLE
				   targRlum,				// red luminance of targets/FP: 0 -> 1
				   targGlum,				// green luminance of targets/FP: 0 -> 1
				   targBlum;
	GLfloat targVertex[3][DRAW_TARG_SLICES*3];
	double density,
		   drawTarget,
		   drawFixationPoint,
		   drawTarget1,
		   drawTarget2,
		   drawBackground,
		   targetSize,
		   luminance,
		   probability,
		   use_lifetime,
		   cutoutRadius;
	int totalStars,
		lifetime,
		drawMode;
	bool useCutout;
} StarField;

// Represents a single star.
typedef struct STAR_STRUCT
{
	// Defines the 3 vertexes necessary to form a triangle.
	GLdouble x[3], y[3], z[3];
} Star;

// Defines a window that contains a wxGLCanvas to display OpenGL stuff.
class GLWindow : public wxFrame
{
private:
	GLPanel *m_glpanel;
	int m_clientX, m_clientY;

public:
	GLWindow(const wxChar *title, int xpos, int ypos, int width, int height,
			 Frustum frustum, StarField starfield);

	// Returns a pointer to the embedded wxGLCanvas.
	GLPanel * GetGLPanel() const;
};


class GLPanel : public wxGLCanvas
{
private:
	Frustum m_frustum;			// Defines the frustum used in glFrustum().
	StarField m_starfield;		// Defines the starfield which will be rendered.
	GLfloat m_Heave,
			m_Surge,
			m_Lateral;
	Star *m_starArray;
	int m_frameCount,
		m_rotationType;
	nm3DDatum m_rotationVector;
	double m_rotationAngle,
		   m_centerX,
		   m_centerY,
		   m_centerZ;
	bool m_doRotation;
	GLuint m_starFieldCallList;

public:
	GLPanel(wxWindow *parent, int width, int height, Frustum frustum, StarField starfield, int *attribList);
	~GLPanel();

	double adjusted_offset;
	double adjusted_ele_offset;
	void OnPaint(wxPaintEvent &event);
	void OnSize(wxSizeEvent &event);
	void SetHeave(GLdouble heave);
	void SetLateral(GLdouble lateral);
	void SetSurge(GLdouble surge);

	// This is the main function that draws the OpenGL scene.
	GLvoid Render();

	// Does any one time OpenGL initializations.
	GLvoid InitGL();

	// Gets/Sets the frustum for the GL scene.
	Frustum * GetFrustum();
	GLvoid SetFrustum(Frustum frustum);

	// Gets/Sets the starfield data for the GL scene and recreates the
	// individual star information.
	StarField * GetStarField();
	GLvoid SetStarField(StarField starfield);

	// Sets the rotation vector.
	void SetRotationVector(nm3DDatum rotationVector);

	// Sets the rotation angle in degrees.
	void SetRotationAngle(double angle);

	// Sets whether or not we should do any rotation.
	void DoRotation(bool val);

	// Sets whether or not we rotate the fixation point, the background, or both.
	// val = 0, rotate the background, but not the fixation point.
	// val = 1, rotate both the background and fixation point.
	// val = 2, rotate the fixation point, but not the background.
	void RotationType(int val);

	// Sets the center of rotation.
	void SetRotationCenter(double x, double y, double z);

	GLvoid DrawTargetObject(int targObj);

private:
	// Calcultates the glFrustum for a stereo scene.
	GLvoid CalculateStereoFrustum(GLfloat screenWidth, GLfloat screenHeight, GLfloat camera2screenDist,
								  GLfloat clipNear, GLfloat clipFar, GLfloat eyeSeparation,
								  GLfloat centerOffsetX, GLfloat centerOffsetY);

	// Generates the starfield.
	GLvoid GenerateStarField();

	// Draws the generated starfield.
	GLvoid DrawStarField();

	// Used to alter star locations due to their lifetimes.
	GLvoid ModifyStarField();

	// Create a glCallList for hollow sphere dots(circle)
	GLvoid SetupCallList();

public:
	// for drawing calibration line of min and max rotation angle
	double minPredictedAngle, maxPredictedAngle;
	double lineWidth, lineLength, squareSize, squareCenter[2];
	int calibTranOn[5], calibRotOn[2], calibRotMotion, calibSquareOn;
	GLfloat *starFieldVertex3D;
	vector <int> dotSizeDistribute;
	int totalDotsNum;
	vector <double> aziVector, eleVector;
	bool redrawCallList;
	double FP_cross[6];
	double hollowSphereRadius;

	void SetupParameters();

private:
	DECLARE_EVENT_TABLE()
};


const double PID2=atan(1.0)*2.0;

#if USING_FISHEYE
bool getWarppedForFishEye(POS3D p, double &u, double &v);
#endif


