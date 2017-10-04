#include "precomp.h" // include (only) this in every .cpp file

TrainingData trainingData( 4000, 1000, 1000 );
Network nn;

// -----------------------------------------------------------
// Initialize the application
// -----------------------------------------------------------
void Game::Init()
{
	FILE* f = fopen( "neuralnet.bin", "rb" );
	if (f)
	{
		// load stored net
		int size = (INPUTSIZE + 1) * (NUMHIDDEN + 1) + (NUMHIDDEN + 1) * NUMOUTPUT;
		float* data = new float[size];
		fread( data, 4, size, f );
		nn.LoadWeights( data );
		fclose( f );
	}
	else
	{
		// train the net using the MNIST set
		printf( "Loading data..." );
		FILE* ff = fopen( "data/images.bin", "rb" );
		for( int i = 0; i < 4000; i++ ) fread( trainingData.trainingSet.entry[i].inputs, 4, 784, ff );
		for( int i = 0; i < 1000; i++ ) fread( trainingData.generalizationSet.entry[i].inputs, 4, 784, ff );
		for( int i = 0; i < 1000; i++ ) fread( trainingData.validationSet.entry[i].inputs, 4, 784, ff );
		fclose( ff );
		ff = fopen( "data/labels.bin", "rb" );
		for( int i = 0; i < 4000; i++ ) fread( trainingData.trainingSet.entry[i].expected, 4, 10, ff );
		for( int i = 0; i < 1000; i++ ) fread( trainingData.generalizationSet.entry[i].expected, 4, 10, ff );
		for( int i = 0; i < 1000; i++ ) fread( trainingData.validationSet.entry[i].expected, 4, 10, ff );
		fclose( ff );
		// create neural network trainer
		nn.Train( trainingData );
		// save the trained net
		int size = (INPUTSIZE + 1) * (NUMHIDDEN + 1) + (NUMHIDDEN + 1) * NUMOUTPUT;
		float* data = new float[size];
		nn.SaveWeights( data );
		FILE* f = fopen( "neuralnet.bin", "wb" );
		fwrite( data, 4, size, f );
		fclose( f );
	}
	// initialize interface
	screen->Clear( 0 );
	int cx = SCRWIDTH / 2, cy = SCRHEIGHT / 2;
	screen->Box( cx - 28 * 6 - 2, cy - 28 * 6 - 2, cx + 28 * 6 + 2, cy + 28 * 6 + 2, 0xffffff );
	buttonDown = false;
}

// -----------------------------------------------------------
// Main application tick function
// -----------------------------------------------------------
void Game::Tick( float deltaTime )
{
	int cx = SCRWIDTH / 2, cy = SCRHEIGHT / 2;
	if (GetAsyncKeyState( VK_LBUTTON ))
	{
		// get mouse position relative to window
		POINT p;
		GetCursorPos( &p );
		ScreenToClient( FindWindow( NULL, TEMPLATE_VERSION ), &p );
		// draw from last pos to current pos
		if (!buttonDown)
		{
			buttonDown = true, lastx = p.x, lasty = p.y;
			screen->Clear( 0 );
			screen->Box( cx - 28 * 6 - 2, cy - 28 * 6 - 2, cx + 28 * 6 + 2, cy + 28 * 6 + 2, 0xffffff );
		}
		int l = (int)sqrtf( (float)(p.x - lastx) * (p.x - lastx) + (p.y - lasty) * (p.y - lasty) ) / 3;
		for( int i = 0; i < l; i++ )
		{
			int tx = lastx + ((p.x - lastx) * i) / l, ty = lasty + ((p.y - lasty) * i) / l;
			for( int y = ty - 7; y <= ty + 7; y++ ) for( int x = tx - 7; x < tx + 7; x++ )
				if (x > (cx - 200) && x < (cx + 200) && (y > (cy - 200) && (y < cy + 200)))
					screen->Plot( x, y, 0xffffff );
		}
		lastx = p.x, lasty = p.y;
	}
	else 
	{
		if (buttonDown)
		{
			// convert image to input for neural net
			float input[28 * 28];
			screen->Box( 2, 2, 33, 33, 0xffffff );
			for( int ty = 0; ty < 28; ty++ ) for( int tx = 0; tx < 28; tx++ ) 
			{
				int total = 0, sx = cx - 28 * 6 + tx * 12, sy = cy - 28 * 6 + ty * 12;
				for( int y = 0; y < 12; y++ ) for( int x = 0; x < 12; x++ )
					total += screen->GetBuffer()[sx + x + (sy + y) * SCRWIDTH] & 255;
				total /= 144;
				input[tx + ty * 28] = (float)(total / 255.0f);
				screen->Plot( 4 + tx, 4 + ty, total + (total << 8) + (total << 16) );
			}
			// query neural net
			const int* result = nn.Evaluate( input );
			float bestScore = -100.0f;
			int best = 0;
			for( int i = 0; i < 10; i++ ) 
			{
				printf( "%4.1f ", nn.outputNeurons[i] );
				if (nn.outputNeurons[i] > bestScore) best = i, bestScore = nn.outputNeurons[i];
			}
			printf( " - is it %i?\n", best );
		}
		buttonDown = false;
	}
}