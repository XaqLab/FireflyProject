#include "StdAfx.h"
#include "ParameterList.h"

// Critical function lock for the parameter list.
wxCriticalSection g_paramLock;

// Global parameter list
CParameterList g_pList;

CParameterList::CParameterList()
{
	LoadHash();
}

void CParameterList::LoadHash()
{
	ParameterValue x;
	int i;


	/***************** Six value parameters *******************/
	x.data.clear();
	x.variable = false;
	for (i = 0; i < 6; i++) {
		x.data.push_back(0.0);
	}
	// Offsets that are added onto the center of rotation.
	x.description = "Offsets added to the center of rotation (x,y,z)cm (Platform/GL).";
	m_pHash.insert(ParameterKeyPair("ROT_CENTER_OFFSETS", x));

	// Fixation Cross setting
    x.data.at(0) = 0.0; //On/Off
	x.data.at(1) = 1.0; //Screen/Edge [0,1]
	x.data.at(2) = 2.0; //line length (cm)
	x.data.at(3) = 3.0; //line width (pixel)
	x.data.at(4) = 0.0; //Shadow [On/Off]
	x.data.at(5) = 7.0; //Shadow line width (pixel)
	x.description = "FP cross [On/Off]; Screen/Edge[0,1]; length(cm); width(pixel); Shadow[On/Off]; Shadow width(pixel)";
	m_pHash.insert(ParameterKeyPair("FP_CROSS", x));

	/***************** Five value parameters *******************/
	x.data.clear();
	x.variable = false;
	for (i = 0; i < 5; i++) {
		x.data.push_back(0.0);
	}
	// Drawing vertical or horizontal or both line of target.
	x.description = "Measure tran amp: [Horizontal,Vertical] lines, ON/OFF=1.0/0.0; location:(x,y,z)";
	m_pHash.insert(ParameterKeyPair("CALIB_TRAN_ON", x));

	/***************** Four value parameters *******************/
	x.data.clear();
	for (i = 0; i < 4; i++) {
		x.data.push_back(0.0);
	}

	// Platform excursions for 2 interval movement.
	x.data.at(0) = 0.1; x.data.at(1) = 0.1; x.data.at(2) = 0.1; x.data.at(3) = 0.1;
	x.description = "2 interval excursions (m)";
	m_pHash.insert(ParameterKeyPair("2I_DIST", x));

	// Duration for each 2 interval movement.
	x.data.at(0) = 2.0; x.data.at(1) = 2.0; x.data.at(2) = 2.0; x.data.at(3) = 2.0;
	x.description = "2 interval duration (s)";
	m_pHash.insert(ParameterKeyPair("2I_TIME", x));

	// Sigma for each 2 interval movement.
	x.data.at(0) = 3.0; x.data.at(1) = 3.0; x.data.at(2) = 3.0; x.data.at(3) = 3.0;
	x.description = "2 interval sigma";
	m_pHash.insert(ParameterKeyPair("2I_SIGMA", x));

	// Elevations for the 2 interval movement.
	x.data.at(0) = 0.0; x.data.at(1) = 0.0; x.data.at(2) = 0.0; x.data.at(3) = 0.0;
	x.description = "2 interval elevation (deg)";
	m_pHash.insert(ParameterKeyPair("2I_ELEVATION", x));

	// Azimuths for the 2 interval movement.
	x.data.at(0) = 90.0; x.data.at(1) = 90.0; x.data.at(2) = 90.0; x.data.at(3) = 90.0;
	x.description = "2 interval azimuth (deg)";
	m_pHash.insert(ParameterKeyPair("2I_AZIMUTH", x));

	// Measure square size on screen.
	x.data.at(0) = 0.0; x.data.at(1) = 10.0; x.data.at(2) = 0.0; x.data.at(3) = 0.0;
	x.description = "Calib Square: [On/Off=1/0,size(cm),center(x,y)]";
	m_pHash.insert(ParameterKeyPair("CALIB_SQUARE", x));

	// Parameters for drawing hollow sphere dots.
	x.data.at(0) = 50.0;	// radius (cm)
	x.data.at(1) = 0.05;	// dots/cm^2
	x.data.at(2) = 3.0;		// Gaussian mean (degree)
	x.data.at(3) = 1.0;		// Gaussian sigma (degree)
	x.description = "Para for dots' sphere surface: radius(cm), density(dots/cm^2), Gauss mean and sigma for dot size.";
	m_pHash.insert(ParameterKeyPair("DOTS_SPHERE_PARA", x));

	// Step Velocity - change Sinusoid rotation movement to constant rotation speed
	x.description = "Step velocity: ";
    x.data.at(0) = 10.0;	// Angular speed (deg/s)
	x.data.at(1) = 300;		// Duration (s)
	x.data.at(2) = 0.0;		// Azimuth (degree)
	x.data.at(3) = 90.0;		// Elevatin (degree)
	x.description = "Step velocity para: Angular speed(deg/s), Duration(s), Azimuth(deg), Elevation(deg).";
    m_pHash.insert(ParameterKeyPair("STEP_VELOCITY_PARA", x));

	/***************** Three value parameters *******************/
	x.data.clear();
	for (i = 0; i < 3; i++) {
		x.data.push_back(0.0);
	}


	x.description = "X-dim(deg) for fixation point, target 1 & 2.";
	x.data[0] = 0.3; x.data[1] = 0.3; x.data[2] = 0.3;
	m_pHash.insert(ParameterKeyPair("TARG_XSIZ", x));

	// Y-dim(deg) for fixation point, target 1 & 2.
	x.description = "Y-dim(deg) for fixation point, target 1 & 2.";
	x.data[0] = 0.3; x.data[1] = 0.3; x.data[2] = 0.3;
	m_pHash.insert(ParameterKeyPair("TARG_YSIZ", x));

	// Shape(0=Ellipse, 1=Rect, 2=Plus, 3=Cross) for fixation point, target 1 & 2.
	x.description = "Shape(0=Ellipse, 1=Rect, 2=Plus, 3=Cross) for fixation point, target 1, 2.";
	x.data[0] = 0.0; x.data[1] = 0.0; x.data[2] = 0.0;
	m_pHash.insert(ParameterKeyPair("TARG_SHAPE", x));

	// Red luminance(0->1) for fixation point, target 1 & 2.
	x.description = "Red luminance(0->1) for fixation point, target 1 & 2.";
	x.data[0] = 1.0; x.data[1] = 0.0; x.data[2] = 0.0;
	m_pHash.insert(ParameterKeyPair("TARG_RLUM", x));

	// Green luminance(0->1) for fixation point, target 1 & 2.
	x.description = "Green luminance(0->1) for fixation point, target 1 & 2.";
	x.data[0] = 1.0; x.data[1] = 0.0; x.data[2] = 0.0;
	m_pHash.insert(ParameterKeyPair("TARG_GLUM", x));

	// Blue luminance(0->1) for fixation point, target 1 & 2.
	x.description = "Blue luminance(0->1) for fixation point, target 1 & 2.";
	x.data[0] = 0.0; x.data[1] = 0.0; x.data[2] = 0.0;
	m_pHash.insert(ParameterKeyPair("TARG_BLUM", x));

	// Default point of origin.
	x.description = "Point of Origin (x, y, z) (m)";
	m_pHash.insert(ParameterKeyPair("M_ORIGIN", x));

	// Global coordinates for the center of the platform.
	x.data.at(0) = 0.0; x.data.at(1) = 0.0; x.data.at(2) = 0.0;
	x.description = "Platform center coordinates.";
	m_pHash.insert(ParameterKeyPair("PLATFORM_CENTER", x));

	// Global coordinates for the center of the head.
	x.data.at(0) = 0.0; x.data.at(1) = 0.0; x.data.at(2) = 0.0;
	x.description = "Center of head based around cube center (cm).";
	m_pHash.insert(ParameterKeyPair("HEAD_CENTER", x));

	// Coordinates for the rotation origin.
	x.data.at(0) = 0.0; x.data.at(1) = 0.0; x.data.at(2) = 0.0;
	x.description = "Rotation origin (yaw, pitch, roll) (deg).";
	m_pHash.insert(ParameterKeyPair("ROT_ORIGIN", x));

	// Starfield dimensions in cm.
	x.description = "Starfield dimensions in cm.";
	x.data[0] = 100.0; x.data[1] = 100.0; x.data[2] = 50.0;
	m_pHash.insert(ParameterKeyPair("STAR_VOLUME", x));

	// X-coordinate for fixation point, target 1 & 2.
	x.description = "Target x-coordinates.";
	x.data[0] = 0.0; x.data[1] = 0.0; x.data[2] = 0.0;
	m_pHash.insert(ParameterKeyPair("TARG_XCTR", x));

	// Y-coordinate for fixation point, target 1 & 2.
	x.description = "Target y-coordinates.";
	x.data[0] = 0.0; x.data[1] = 0.0; x.data[2] = 0.0;
	m_pHash.insert(ParameterKeyPair("TARG_YCTR", x));

	// Z-coordinate for fixation point, target 1 & 2.
	x.description = "Target z-coordinates.";
	x.data[0] = 0.0; x.data[1] = 0.0; x.data[2] = 0.0;
	m_pHash.insert(ParameterKeyPair("TARG_ZCTR", x));

	// Z-coordinate for fixation point, target 1 & 2.
	x.description = "Target z-coordinates.";
	x.data[0] = 1.0; x.data[1] = 1.0; x.data[2] = 1.0;
	m_pHash.insert(ParameterKeyPair("TARG_LUM_MULT", x));

	// Color for the left star scene.
	x.description = "Color for the left-eye stars.";
	x.data[0] = 1.0; x.data[1] = 0.0; x.data[2] = 0.0;
	m_pHash.insert(ParameterKeyPair("STAR_LEYE_COLOR", x));

	// Color for the right star scene.
	x.description = "Color for the right-eye stars.";
	x.data[0] = 0.0; x.data[1] = 1.0; x.data[2] = 0.0;
	m_pHash.insert(ParameterKeyPair("STAR_REYE_COLOR", x));

	// Magnitude for the noise.
	x.description = "Noise magnitude (std deviations).";
	x.data[0] = 0.01; x.data[1] = 0.01; x.data[2] = 0.01;
	m_pHash.insert(ParameterKeyPair("NOISE_MAGNITUDE", x));

	// Location of the monocular eye measured from the center of the head.
	x.data.at(0) = 0.0; x.data.at(1) = 0.0; x.data.at(2) = 0.0;
	x.description = "Location of the monocular eye measured from the center of the head. (x, y, z)cm.";
	m_pHash.insert(ParameterKeyPair("EYE_OFFSETS", x));

	/***************** Two value parameters *****************/
	x.data.clear();
	for (i = 0; i < 2; i++) {
		x.data.push_back(0.0);
	}

	// Delay between the 1st and 2nd 2I movement.
	x.description = "Delay between the 2I movement (s).";
	x.data.at(0) = 0.0; x.data.at(1) = 0.0;
	m_pHash.insert(ParameterKeyPair("2I_DELAY", x));

	// Screen dimensions.
	x.description = "Screen Width and Height (cm).";
	x.data[0] = 58.8;	// Width 
	x.data[1] = 58.9;	// Height
	m_pHash.insert(ParameterKeyPair("SCREEN_DIMS", x));

	// Near and far clipping planes.
	x.description = "Near and Far clipping planes (cm).";
	x.data[0] = 5.0;	// Near
	x.data[1] = 150.0;	// Far
	m_pHash.insert(ParameterKeyPair("CLIP_PLANES", x));

	// Triangle dimensions.
	x.description = "Triangle Base and Height (cm).";
#if DEBUG_DEFAULTS
	x.data[0] = 0.3; x.data[1] = 0.3;
#else
	x.data[0] = 1.15; x.data[1] = 1.15;
#endif
	m_pHash.insert(ParameterKeyPair("STAR_SIZE", x));

	// Elevation
	x.description = "Elevation in degrees (Base/GL).";
	x.data[0] = -90.0; x.data[1] = -90.0;
	m_pHash.insert(ParameterKeyPair("M_ELEVATION", x));

	// Azimuth
	x.description = "Azimuth in degrees (Base/GL).";
	x.data[0] = 0.0; x.data[1] = 0.0;
	m_pHash.insert(ParameterKeyPair("M_AZIMUTH", x));

	// Noise Elevation
	x.description = "Noise Elevation in degrees (Base/GL).";
	x.data[0] = 0.0; x.data[1] = 0.0;
	m_pHash.insert(ParameterKeyPair("NOISE_ELEVATION", x));

	// Noise Azimuth
	x.description = "Noise Azimuth in degrees (Base/GL).";
	x.data[0] = 0.0; x.data[1] = 0.0;
	m_pHash.insert(ParameterKeyPair("NOISE_AZIMUTH", x));

	// Distance travelled my the motion base.
	x.description = "Movement Magnitude (m) (Base/GL).";
	x.data[0] = 0.10; x.data[1] = 0.10;
	m_pHash.insert(ParameterKeyPair("M_DIST", x));

	// Default movement duration.
	x.data[0] = 2.0; x.data[1] = 2.0;
	x.description = "Movement Duration x (s) (Base/GL).";
	m_pHash.insert(ParameterKeyPair("M_TIME", x));

	// Number of sigmas in the Gaussian.
	x.data[0] = 3.0; x.data[1] = 3.0;
	x.description = "Number of sigmas in the Gaussian (Base/GL).";
	m_pHash.insert(ParameterKeyPair("M_SIGMA", x));

	/*
	// Number of cycles of sine wave in the Gabor.
	x.data[0] = 1.0; x.data[1] = 1.0;
	x.description = "Number of sine wave in the Gabor (Base/GL).";
	m_pHash.insert(ParameterKeyPair("M_CYCLE", x));
	*/

	// Elevation of the axis of rotation.
#if DEBUG_DEFAULTS
	x.data.at(0) = 0.0;
	x.data.at(1) = 0.0;
#else
	x.data.at(0) = -90.0;
	x.data.at(1) = -90.0;
#endif
	x.description = "Axis of rotation elevation.";
	m_pHash.insert(ParameterKeyPair("ROT_ELEVATION", x));

	// Azimuth of the axis of rotation.
#if DEBUG_DEFAULTS
	x.data.at(0) = 0.0;
	x.data.at(1) = 0.0;
#else
	x.data.at(0) = 0.0;
	x.data.at(1) = 0.0;
#endif
	x.description = "Axis of rotation azimuth.";
	m_pHash.insert(ParameterKeyPair("ROT_AZIMUTH", x));

	// Amplitude of rotation.
#if DEBUG_DEFAULTS
	x.data.at(0) = 5.0;
	x.data.at(1) = 5.0;
#else
	x.data.at(0) = 5.0;
	x.data.at(1) = 5.0;
#endif
	x.description = "Amplitude of rotation.";
	m_pHash.insert(ParameterKeyPair("ROT_AMPLITUDE", x));

	// Duration of rotation.
#if DEBUG_DEFAULTS
	x.data.at(0) = 2.0;
	x.data.at(1) = 2.0;
#else
	x.data.at(0) = 2.0;
	x.data.at(1) = 2.0;
#endif
	x.description = "Duration of rotation.";
	m_pHash.insert(ParameterKeyPair("ROT_DURATION", x));

	// Number of sigmas in the Gaussian rotation.
#if DEBUG_DEFAULTS
	x.data.at(0) = 6.0;
	x.data.at(1) = 6.0;
#else
	x.data.at(0) = 3.0;
	x.data.at(1) = 3.0;
#endif
	x.description = "Number of sigmas in rot Gaussian.";
	m_pHash.insert(ParameterKeyPair("ROT_SIGMA", x));

	// Phase of rotation.
	x.data.at(0) = 0.0;
	x.data.at(1) = 0.0;
	x.description = "Phase of rotation (deg).";
	m_pHash.insert(ParameterKeyPair("ROT_PHASE", x));

	// Amplitude of the translational sinusoid (m).
	x.data.at(0) = 0.05; x.data.at(1) = 0.05;
	x.description = "Sinusoid translational amplitude (m).";
	m_pHash.insert(ParameterKeyPair("SIN_TRANS_AMPLITUDE", x));

	// Amplitude of the rotational sinusoid (deg).
	x.data.at(0) = 5.0; x.data.at(1) = 5.0; 
	x.description = "Sinusoid rotational amplitude (deg).";
	m_pHash.insert(ParameterKeyPair("SIN_ROT_AMPLITUDE", x));

	// Frequency of the sinusoid (Hz).
	x.data.at(0) = 0.5; x.data.at(1) = 0.5;
	x.description = "Sinusoid frequency (Hz)";
	m_pHash.insert(ParameterKeyPair("SIN_FREQUENCY", x));

	// Elevation of the sinusoid (deg).
	x.data.at(0) = 0.0; x.data.at(1) = 0.0; 
	x.description = "Sinusoid elevation (deg)";
	m_pHash.insert(ParameterKeyPair("SIN_ELEVATION", x));

	// Azimuth of the sinusoid (deg).
	x.data.at(0) = 0.0; x.data.at(1) = 0.0;
	x.description = "Sinusoid azimuth (deg)";
	m_pHash.insert(ParameterKeyPair("SIN_AZIMUTH", x));

	// Duration of the sinusoid (s)
	x.data.at(0) = 60.0*5.0; x.data.at(1) = 60*5.0;
	x.description = "Sinusoid duration (s)";
	m_pHash.insert(ParameterKeyPair("SIN_DURATION", x));

	// Normalization factors for the CED analog output.
	x.description = "Normalization factors [min, max].";
	x.data.at(0) = 0.0; x.data.at(1) = 1.0;
	m_pHash.insert(ParameterKeyPair("STIM_NORM_FACTORS", x));

	// Help us to measure the amplitude of sinusoid rotation motion
	x.description = "Measure rot amp: [Fixed, Move] lines, ON/OFF=1.0/0.0";
	x.data.at(0) = 0.0; x.data.at(1) = 0.0;
	m_pHash.insert(ParameterKeyPair("CALIB_ROT_ON", x));

	// Help us to measure the amplitude of sinusoid rotation motion
	x.description = "Measurement line setup: width(pixel) and length(cm)";
	x.data.at(0) = 2.0; x.data.at(1) = 10.0;
	m_pHash.insert(ParameterKeyPair("CALIB_LINE_SETUP", x));

	x.data.at(0)=20.0;x.data.at(1)=10.0;
	x.description = "Offset For Adjustment(1=origin, 2=elevation.";
	m_pHash.insert(ParameterKeyPair("ADJUSTED_OFFSET", x));

	/***************** One value parameters *****************/
	x.data.clear();
	x.data.push_back(0.0);

	// Flags a circle cutout to appear at the center of the screen.
#if DEBUG_DEFAULTS
	x.data.at(0) = 0.0;
#else
	x.data.at(0) = 0.0;
#endif
	x.description = "Flags circle cutout.";
	m_pHash.insert(ParameterKeyPair("USE_CUTOUT", x));

	// Setup condition for output velocity curve to CED
	x.description = "Output velocity curve to CED. true=1.0,fales=0.0";
	x.data[0] = 0.0;
	m_pHash.insert(ParameterKeyPair("OUTPUT_VELOCITY", x));

	// Output to D/A board of the stumulus; sent to the Moog and the frame delayed opengl singal.
	x.description = "Signal output to D/A board. [Lateral,Heave,Surge,Yaw,Pitch,Roll]=[1,2,3,4,5,6]";
	x.data[0] = 0.0;
	m_pHash.insert(ParameterKeyPair("STIM_ANALOG_OUTPUT", x));

	// Setup condition for output amplitude curve to CED
	x.description = "Mult. of Signal output to D/A board";
	x.data[0] = 1.0;
	m_pHash.insert(ParameterKeyPair("STIM_ANALOG_MULT", x));

	// The radius of the cutout.
	x.data.at(0) = 5.0;
	x.description = "Cutout radius (cm)";
	m_pHash.insert(ParameterKeyPair("CUTOUT_RADIUS", x));

	// Determines if sinusoidal motion is continuous.
	x.data.at(0) = 0.0; 
	x.description = "Sin continuous.  0 = no, 1 = yes";
	m_pHash.insert(ParameterKeyPair("SIN_CONTINUOUS", x));

	// Sets whether rotation is translational or rotational.
	x.data.at(0) = 0.0; 
	x.description = "Sin Mode.  0 = trans, 1 = rot";
	m_pHash.insert(ParameterKeyPair("SIN_MODE", x));

#if DEBUG_DEFAULTS
	x.data.at(0) = 1.0;
#else
	x.data.at(0) = 0.0;
#endif
	x.description = "Toggle fixation point rotation.";
	m_pHash.insert(ParameterKeyPair("FP_ROTATE", x));

	// Delay after the sync.
	x.data.at(0) = 0.0;
	x.description = "Delay (ms) after the sync.";
	m_pHash.insert(ParameterKeyPair("SYNC_DELAY", x));

	// Mode for the tilt/translation.
	x.data.at(0) = 0.0;
	x.description = "Mode for t/t.";
	m_pHash.insert(ParameterKeyPair("TT_MODE", x));

	// Sets how many dimensions of noise we want to use.
	x.description = "Number of noise dimensions (1,2, or 3).";
	x.data[0] = 1.0;
	m_pHash.insert(ParameterKeyPair("NOISE_DIMS", x));

	// Determines whether or not we multiply the noise by a high
	// powered Gaussian to make the ends zero.
	x.description = "Fix noise (1=yes,0=no).";
	x.data[0] = 1.0;
	m_pHash.insert(ParameterKeyPair("FIX_NOISE", x));

	// Determines if we multiply the filter by a high powered
	// Gaussian.
	x.description = "Fix filter (1=yes,0=no).";
	x.data[0] = 1.0;
	m_pHash.insert(ParameterKeyPair("FIX_FILTER", x));

	// Filter frequency.
	x.description = "Filter frequency (.1-10Hz).";
	x.data[0] = 5.0;
	m_pHash.insert(ParameterKeyPair("CUTOFF_FREQ", x));

	// Gaussian normal distribution seed.
	x.description = "Gaussian normal distribution seed > 0";
	x.data[0] = 1978.0;
	m_pHash.insert(ParameterKeyPair("GAUSSIAN_SEED", x));

	// Star lifetime.
	x.description = "Star lifetime (#frames).";
	x.data[0] = 5.0;
	m_pHash.insert(ParameterKeyPair("STAR_LIFETIME", x));

	// Star motion coherence factor.  0 means all stars change.
	x.description = "Star motion coherence (% out of 100).";
	x.data[0] = 15.0;
	m_pHash.insert(ParameterKeyPair("STAR_MOTION_COHERENCE", x));
	
	// Turns star lifetime on and off.
	x.description = "Star lifetime on/off.";
	x.data[0] = 0.0;
	m_pHash.insert(ParameterKeyPair("STAR_LIFETIME_ON", x));

	// Star luminance multiplier.
	x.description = "Star luminance multiplier.";
	x.data[0] = 1.0;
	m_pHash.insert(ParameterKeyPair("STAR_LUM_MULT", x));

	// Target 1 on/off.
	x.description = "Target 1 on/off.";
	x.data[0] = 0.0;
	m_pHash.insert(ParameterKeyPair("TARG1_ON", x));

	// Target 2 on/off.
	x.description = "Target 2 on/off.";
	x.data[0] = 0.0;
	m_pHash.insert(ParameterKeyPair("TARG2_ON", x));
	
	// Fixation point on/off.
	x.description = "Fixation point on/off.";
#if DEBUG_DEFAULTS
	x.data[0] = 1.0;
#else
	x.data[0] = 0.0; 
#endif
	m_pHash.insert(ParameterKeyPair("FP_ON", x));

	// Decides if noise should be added.
	x.description = "Enables noise. (0=off, 1=on)";
	x.data[0] = 0.0;
	m_pHash.insert(ParameterKeyPair("USE_NOISE", x));

	// Turns movement on and off.
	x.description = "Enables motion base movement. (0.0=off, 1.0==on)";
	x.data[0] = 0.0;
	m_pHash.insert(ParameterKeyPair("DO_MOVEMENT", x));

	x.description = "Makes the motion base move to zero position.";
	x.data[0] = 0.0;
	m_pHash.insert(ParameterKeyPair("GO_TO_ZERO", x));


	// The amount of offset given to a library trajectory.
	// This value will reset in MoogDots::InitRig().
	x.description = "Offset used to shift predicted trajectory (ms).";
	x.data.at(0) = 500.0;
	m_pHash.insert(ParameterKeyPair("PRED_OFFSET", x));

	// The amount of offset given to frame delay of different projectors,
	// so that we can generate a same signal as accelerometer.
	// This value will reset in MoogDots::InitRig().
	x.description = "Offset used to adjust frame delay of different projectors (ms).";
	x.data.at(0) = 16.0;
	m_pHash.insert(ParameterKeyPair("FRAME_DELAY", x));

	/*
	// Indicates if the target should be on.
	x.description = "Indicates if the target should be on.";
	x.data[0] = 0.0;
	m_pHash.insert(ParameterKeyPair("TARG_CROSS", x));
	*/

	// Size of the center target.
	x.description = "Size of the center target.";
	x.data[0] = 7.0;
	m_pHash.insert(ParameterKeyPair("TARGET_SIZE", x));

	// Indicates if the background shoud be on.
	x.description = "Indicates if the background should be on.";
#if DEBUG_DEFAULTS
	x.data[0] = 1.0;
#else
	x.data[0] = 0.0; 
#endif
	m_pHash.insert(ParameterKeyPair("BACKGROUND_ON", x));

	// Center of dots' sphere is at monkey eye, but center of triangles' or dots' cube at screen.
	x.description = "Draw volume: 0)dots' soild sphere; 1)triangles' cube; 2)dots' cube; 3) dots' hollow sphere";
	x.data[0] = 1.0; 
	m_pHash.insert(ParameterKeyPair("DRAW_MODE", x));

	// Draw dots size.
	x.description = "Draw dots size.";
	x.data[0] = 4.0;
	m_pHash.insert(ParameterKeyPair("DOTS_SIZE", x));


	// Motion type.
	x.description = "Motion Type: 0=Gaussian, 1=Rotation, 2=Sinusoid, 3=2I, 4=TT, 5=Gabor, 6=Step Velocity";
#if DEBUG_DEFAULTS
	x.data[0] = 0.0;
#else
	x.data[0] = 0.0; 
#endif
	m_pHash.insert(ParameterKeyPair("MOTION_TYPE", x));

	// Interocular distance.
	x.description = "Interocular distance (cm).";
#if DEBUG_DEFAULTS
	x.data.at(0) = 0.7;
#else
	x.data[0] = 6.5;
#endif
	m_pHash.insert(ParameterKeyPair("IO_DIST", x));

	// Starfield density (stars/cm^3).
	x.description = "Starfield Density (stars/cm^3).";
	x.data[0] = 0.01;
	m_pHash.insert(ParameterKeyPair("STAR_DENSITY", x));

	// The amount of offset given to analog output.
    x.description = "Offset used to shift predicted stimulus (ms)";
    x.data.at(0) = 465.00;
    m_pHash.insert(ParameterKeyPair("PRED_OFFSET_STIMULUS", x));

	// Help us to measure the amplitude of sinusoid rotation motion
	x.description = "Measure rot amp: 0=Pitch, 1=Yaw, 2=Roll";
	x.data.at(0) = 1.0;
	m_pHash.insert(ParameterKeyPair("CALIB_ROT_MOTION", x));

	// OpengGL refresh is synchronized to monitor freq/refresh.
	// as opposed to Moog communication freq.
    x.description = "Enable OpengGL refresh sync to monitor freq.";
    x.data.at(0) = 0.0;
    m_pHash.insert(ParameterKeyPair("VISUAL_SYNC", x));

	// OpengGL refresh is synchronized to monitor freq/refresh.
	// as opposed to Moog communication freq.
    x.description = "Rotation angle.";
    x.data.at(0) = 0.0;
    m_pHash.insert(ParameterKeyPair("ROT_ANGLE", x));
}

void CParameterList::SetVectorData(string key, vector<double> value)
{
	g_paramLock.Enter();

	// Find the key, value pair associated with the given key.
	ParameterIterator i = m_pHash.find(key);

	// Set the key value if we found the key pair.
	if (i != m_pHash.end()) {
		i->second.data = value;
	}

	g_paramLock.Leave();
}

vector<double> CParameterList::GetVectorData(string key)
{
	g_paramLock.Enter();

	vector<double> value;

	// Try to find the pair associated with the given key.
	ParameterIterator i = m_pHash.find(key.c_str());

	// If we found an entry associated with the key, store the data
	// vector associated with it.
	if (i != m_pHash.end()) {
		value = i->second.data;
	}

	g_paramLock.Leave();

	return value;
}

string CParameterList::GetParamDescription(string param)
{
	g_paramLock.Enter();

	string s = "";

	// Find the parameter iterator.
	ParameterIterator i = m_pHash.find(param);

	if (i == m_pHash.end()) {
		s = "No Data Found";
	}
	else {
		s = i->second.description;
	}

	g_paramLock.Leave();

	return s;
}

string * CParameterList::GetKeyList(int &keyCount)
{
	g_paramLock.Enter();

	string *keyList;
	int i;

	// Number of elements in the hash.
	keyCount = m_pHash.size();

	// Initialize the key list.
	keyList = new string[keyCount];

	// Iterate through the hash and extract all the key names.
	ParameterIterator x;
	i = 0;
	for (x = m_pHash.begin(); x != m_pHash.end(); x++) {
		keyList[i] = x->first;
		i++;
	}

	g_paramLock.Leave();

	return keyList;
}

int CParameterList::GetListSize() const
{
	g_paramLock.Enter();
	int hashSize = m_pHash.size();
	g_paramLock.Leave();

	return hashSize;
}


bool CParameterList::IsVariable(string param)
{
	bool isVariable = false;

	g_paramLock.Enter();

	// Try to find the pair associated with the given key.
	ParameterIterator i = m_pHash.find(param.c_str());

	if (i != m_pHash.end()) {
		isVariable = i->second.variable;
	}

	g_paramLock.Leave();

	return isVariable;
}

bool CParameterList::Exists(string key)
{
	bool keyExists = false;

	g_paramLock.Enter();

	// Try to find the pair associated with the given key.
	ParameterIterator i = m_pHash.find(key.c_str());

	if (i != m_pHash.end()) {
		keyExists = true;
	}

	g_paramLock.Leave();

	return keyExists;
}

int CParameterList::GetParamSize(string param)
{
	g_paramLock.Enter();

	int paramSize = 0;

	// Try to find the pair associated with the given key.
	ParameterIterator i = m_pHash.find(param.c_str());

	if (i != m_pHash.end()) {
		paramSize = static_cast<int>(i->second.data.size());
	}

	g_paramLock.Leave();

	return paramSize;
}