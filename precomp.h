// add your includes to this file instead of to individual .cpp files
// to enjoy the benefits of precompiled headers:
// - fast compilation
// - solve issues with the order of header files once (here)
// do not include headers in header files (ever).

#define SCRWIDTH		800
#define SCRHEIGHT		512
// #define FULLSCREEN
// #define ADVANCEDGL	// faster if your system supports it

// neural network settings
#define INPUTSIZE		(28*28)
#define NUMHIDDEN		150
#define NUMOUTPUT		10
#define ORIGINALACTIVATIONFUNCTION // use 1/(1+exp(x)); alternative is faster and more accurate
#define LEARNINGRATE	0.001f
#define MOMENTUM		0.9f
#define MAXEPOCHS		600
#define TARGETACCURACY	95

#include <inttypes.h>
extern "C" 
{ 
#include "glew.h" 
}
#include "gl.h"
#include "io.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include "fcntl.h"
#include "SDL.h"
#include "wglext.h"
#include "freeimage.h"
#include "math.h"
#include "stdlib.h"
#include "emmintrin.h"
#include "immintrin.h"
#include "windows.h"
#include "template.h"
#include "surface.h"
#include "threads.h"
#include <ctime>
#include <random>

using namespace std;
using namespace Tmpl8;

#include "NN\NeuralNetwork.h"
#include "game.h"