//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// Based on Bobby Anguelov's code
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include "../precomp.h"

namespace Tmpl8 {

Network::Network()
{
	// initialize neural net
	InitializeNetwork();
	InitializeWeights();
	// initialize trainer (calloc: malloc + clear to zero)
#if SIMD != OFF
	deltaInputHidden[(INPUTSIZE + 1) * (NUMHIDDEN + 1) + 1] = { 0.f };
	deltaHiddenOutput[simdNumHidden * NUMOUTPUT] = { 0.f };
	errorGradientsHidden[simdNumHidden] = { 0.f };
	errorGradientsOutput[simdNumOutput] = { 0.f };
#else
	deltaInputHidden = (float*)calloc((INPUTSIZE + 1) * (NUMHIDDEN + 1), sizeof(float));
	deltaHiddenOutput = (float*)calloc((NUMHIDDEN + 1) * NUMOUTPUT, sizeof(float));
	errorGradientsHidden = (float*)calloc(NUMHIDDEN + 1, sizeof(float));
	errorGradientsOutput = (float*)calloc(NUMOUTPUT, sizeof(float));
#endif
}

void Network::InitializeNetwork()
{
	// Create storage and initialize the neurons and the outputs
	// Add bias neurons
	const int totalNumInputs = INPUTSIZE + 1, totalNumHiddens = NUMHIDDEN + 1;
	memset( inputNeurons, 0, INPUTSIZE * 4 );
	memset( hiddenNeurons, 0, simdNumOutput * 4 );
	memset( outputNeurons, 0, simdNumOutput * 4 );
	memset( clampedOutputs, 0, simdNumOutput * 4 );
	// Set bias values
	inputNeurons[INPUTSIZE] = hiddenNeurons[NUMHIDDEN] = -1.0f;
	// Create storage and initialize and layer weights
#if SIMD == OFF
	weightsInputHidden = new float[totalNumInputs * totalNumHiddens];
	weightsHiddenOutput = new float[totalNumHiddens * NUMOUTPUT];
#else
	weightsHiddenOutput[simdNumHidden * NUMOUTPUT] = { 0.0f };
	weightsInputHidden[(INPUTSIZE + 1) * (NUMHIDDEN + 1) + 1] = { 0.0f };
#endif
}

void Network::InitializeWeights()
{
	random_device rd;
	mt19937 generator( rd() );
	const float distributionRangeHalfWidth = 2.4f / INPUTSIZE;
	const float standardDeviation = distributionRangeHalfWidth * 2 / 6;
	normal_distribution<> normalDistribution( 0, standardDeviation );
	// set weights to normally distributed random values between [-2.4 / numInputs, 2.4 / numInputs]
	for( int i = 0; i <= INPUTSIZE; i++ ) for( int j = 0; j < NUMHIDDEN; j++ )
	{
		const int weightIdx = GetInputHiddenWeightIndex( i, j );
		weightsInputHidden[weightIdx] = (float)normalDistribution( generator );
	}
	// set weights to normally distributed random values between [-2.4 / numInputs, 2.4 / numInputs]
	for( int i = 0; i <= NUMHIDDEN; i++ ) for( int j = 0; j < NUMOUTPUT; j++ )
	{
		const int weightIdx = GetHiddenOutputWeightIndex( i, j );
		weightsHiddenOutput[weightIdx] = (float)normalDistribution( generator );
	}
}

void Network::LoadWeights( const float* weights )
{
	const int numInputHiddenWeights = (INPUTSIZE + 1) * (NUMHIDDEN + 1);
	const int numHiddenOutputWeights = (NUMHIDDEN + 1) * NUMOUTPUT;
	int weightIdx = 0;
	for( int i = 0; i < numInputHiddenWeights; i++ ) weightsInputHidden[i] = weights[weightIdx++];
	for( int i = 0; i < numHiddenOutputWeights; i++ ) weightsHiddenOutput[i] = weights[weightIdx++];
}

void Network::SaveWeights( float* weights )
{
	const int numInputHiddenWeights = (INPUTSIZE + 1) * (NUMHIDDEN + 1);
	const int numHiddenOutputWeights = (NUMHIDDEN + 1) * NUMOUTPUT;
	int weightIdx = 0;
	for( int i = 0; i < numInputHiddenWeights; i++ ) weights[weightIdx++] = weightsInputHidden[i];
	for( int i = 0; i < numHiddenOutputWeights; i++ ) weights[weightIdx++] = weightsHiddenOutput[i];
}

float Network::GetHiddenErrorGradient( int hiddenIdx ) const
{
	// get sum of hidden->output weights * output error gradients
	float weightedSum = 0;
	for( int i = 0; i < NUMOUTPUT; i++ )
	{
		const int weightIdx = GetHiddenOutputWeightIndex( hiddenIdx, i );
		weightedSum += weightsHiddenOutput[weightIdx] * errorGradientsOutput[i];
	}
	// return error gradient
	return hiddenNeurons[hiddenIdx] * (1.0f - hiddenNeurons[hiddenIdx]) * weightedSum;
}

void Network::Train( const TrainingData& trainingData )
{
	// reset training state
	currentEpoch = 0;
	trainingSetAccuracy = validationSetAccuracy = generalizationSetAccuracy = 0;
	trainingSetMSE = validationSetMSE = generalizationSetMSE = 0;
	// print header
	printf( " Neural Network Training Starting: \n" );
	printf( "==========================================================================\n" );
	printf( " LR: %f, momentum: %f, max epochs: %i\n", LEARNINGRATE, MOMENTUM, MAXEPOCHS );
	printf( " %i input neurons, %i hidden neurons, %i output neurons\n", INPUTSIZE, NUMHIDDEN, NUMOUTPUT );
	printf( "==========================================================================\n" );
	// train network using training dataset for training and generalization dataset for testing
	float allTime = 0.f;
	while ((trainingSetAccuracy < TARGETACCURACY || generalizationSetAccuracy < TARGETACCURACY) && currentEpoch < MAXEPOCHS)
	{
		// use training set to train network
		timer t;
		t.reset();
		RunEpoch( trainingData.trainingSet );
		float epochTime = t.elapsed();
		// get generalization set accuracy and MSE
		GetSetAccuracyAndMSE( trainingData.generalizationSet, generalizationSetAccuracy, generalizationSetMSE );
		allTime += epochTime;
		float avg = allTime / (currentEpoch + 1);
		printf( "Epoch: %03i - TS accuracy: %4.1f, MSE: %4.4f GS accuracy: %4.1f, in %06.1fms (Avg: %06.1fms Speed-up: %.1fx)\n", 
			   currentEpoch, trainingSetAccuracy, trainingSetMSE, generalizationSetAccuracy, epochTime , avg, REFSPEED/avg);
		currentEpoch++;
	}
	// get validation set accuracy and MSE
	GetSetAccuracyAndMSE( trainingData.validationSet, validationSetAccuracy, validationSetMSE );
	// print validation accuracy and MSE
	printf( "\nTraining complete. Epochs: %i\n", currentEpoch );
	printf( " Validation set accuracy: %f\n Validation set MSE: %f\n", validationSetAccuracy, validationSetMSE );
}

void Network::RunEpoch( const TrainingSet& set )
{
	float incorrectEntries = 0, MSE = 0;
	//printf("%d,%d,%d,%d,%d\n", simdNumHidden, simdNumInput, simdNumOutput, simdNumWeightsIH, (INPUTSIZE + 1) * (NUMHIDDEN + 1) + 1);
	for( int i = 0; i < set.size; i++ )
	{
		const TrainingEntry& entry = set.entry[i];
		// Feed inputs through network and back propagate errors
		// Vectorized evaluate
		Evaluate( entry.inputs );
		// Vectorized back-propagate
		Backpropagate( entry.expected );

		// Check all outputs from neural network against desired values
		bool resultCorrect = true;
		for( int j = 0; j < NUMOUTPUT; j++ )
		{
			if (clampedOutputs[j] != entry.expected[j]) resultCorrect = false;
			const float delta = outputNeurons[j] - entry.expected[j];
			MSE += delta * delta;
		}
		if (!resultCorrect) incorrectEntries++;
	}
	// Update training accuracy and MSE
	trainingSetAccuracy = 100.0f - (incorrectEntries / set.size * 100.0f);
	trainingSetMSE = MSE / (NUMOUTPUT * set.size);
}

void Network::Backpropagate(const int* expectedOutputs)
{
#ifdef VERIFY
	float deltaHiddenOutputUnmodified[simdNumHidden*NUMOUTPUT];
	memcpy(&deltaHiddenOutputUnmodified, &deltaHiddenOutput, simdNumHidden*NUMOUTPUT * 4);
	// modify deltas between hidden and output layers
	for (int i = 0; i < NUMOUTPUT; i++)
	{
		// get error gradient for every output node
		errorGradientsOutput[i] = GetOutputErrorGradient((float)expectedOutputs[i], outputNeurons[i]);
		// for all nodes in hidden layer and bias neuron
		for (int j = 0; j <= NUMHIDDEN; j++)
		{
			const int weightIdx = GetHiddenOutputWeightIndex(j, i);
			// calculate change in weight
			deltaHiddenOutput[weightIdx] = LEARNINGRATE * hiddenNeurons[j] * errorGradientsOutput[i] + MOMENTUM * deltaHiddenOutput[weightIdx];
		}
	}
	float deltaHiddenOutputTrue[simdNumHidden*NUMOUTPUT];
	memcpy(&deltaHiddenOutputTrue, &deltaHiddenOutput, simdNumHidden*NUMOUTPUT * 4);
	memcpy(&deltaHiddenOutput, &deltaHiddenOutputUnmodified, simdNumHidden*NUMOUTPUT * 4);

#endif
#if SIMD & VECTORIZE_BACKPROPAGATE
	const __mVec onesVec = _mm_set1_ps(1.f);
	/*for (int i = 0; i < NUMOUTPUT; i++) // Too few to bother in SSE (not to mention AVX), also int to float -> conversion
	{
		errorGradientsOutput[i] = GetOutputErrorGradient((float)expectedOutputs[i], outputNeurons[i]);
	}*/
	//union { __declspec(align(ALIGNMENT)) float expectedOut_[simdNumOutput]; __mVec expectedOutVec[simdNumOutput / VEC_LENGTH]; };
	
	// Error gradient output calculation with SIMD maybe not worth it - it's a minor performance improvement if at all
	for (int i = 0; i < NUMOUTPUT; i++) // Too few to bother
	{
		errorGradientsOutput[i] = GetOutputErrorGradient((float)expectedOutputs[i], outputNeurons[i]);
	}
	/*for (int i = 0; i < simdNumOutput / VEC_LENGTH; i++)
	{
		expectedOutVec[i] = _mVec_setr_ps((float)expectedOutputs[i*VEC_LENGTH], (float)expectedOutputs[i*VEC_LENGTH + 1], (float)expectedOutputs[i*VEC_LENGTH + 2],
											(float)expectedOutputs[i*VEC_LENGTH + 3], (float)expectedOutputs[i*VEC_LENGTH+4], (float)expectedOutputs[i*VEC_LENGTH+5],
											(float)expectedOutputs[i*VEC_LENGTH+6], (float)expectedOutputs[i*VEC_LENGTH+7]);
		__mVec outputNeuronsVec = _mm_load_ps(&outputNeurons[i * VEC_LENGTH]);
		errorGradientsOutputVec[i] = _mm_mul_ps(outputNeuronsVec, _mm_mul_ps(_mm_sub_ps(onesVec, outputNeuronsVec), _mm_sub_ps(expectedOutVec[i], outputNeuronsVec)));
	}*/

	const __mVec learningRateVec = _mm_set1_ps(LEARNINGRATE);
	const __mVec momentumVec = _mm_set1_ps(MOMENTUM);
#if SIMD != OFF && defined(SIMD_OPTIMIZED_SECIAL_CASE_BACKPROP)
	// Extend error gradient output to repeat itself - extend array to 20 size so it gets 5 vectors and fix the overlap
	memcpy(&errorGradientsOutput[NUMOUTPUT], &errorGradientsOutput[0], NUMOUTPUT * sizeof(float));
	// Get error gradient for every output node
	int index = 0;
	// 20 on each turn (simdNumHidden * NUMOUTPUT) 
	for (int j = 0; j < (simdNumHidden * NUMOUTPUT) / VEC_LENGTH; j += 5) // Optimized to length of 4
	{
		__mVec hd4 = _mm_mul_ps(learningRateVec, _mm_set_ps1(hiddenNeurons[index]));
		__mVec hd42 = _mVec_setr_ps(hiddenNeurons[index + 1], hiddenNeurons[index + 1], hiddenNeurons[index], hiddenNeurons[index],0,0,0,0);
		__mVec hd41 = _mm_mul_ps(learningRateVec, _mm_set_ps1(hiddenNeurons[index + 1]));
		deltaHiddenOutputVec[j] = _mm_add_ps(_mm_mul_ps(hd4, errorGradientsOutputVec[0]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec[j]));
		deltaHiddenOutputVec[j + 1] = _mm_add_ps(_mm_mul_ps(hd4, errorGradientsOutputVec[1]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec[j + 1]));
		// Edge case with overlap
		deltaHiddenOutputVec[j + 2] = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(learningRateVec, hd42), errorGradientsOutputVec[2]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec[j + 2]));
		deltaHiddenOutputVec[j + 3] = _mm_add_ps(_mm_mul_ps(hd41, errorGradientsOutputVec[3]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec[j + 3]));
		deltaHiddenOutputVec[j + 4] = _mm_add_ps(_mm_mul_ps(hd41, errorGradientsOutputVec[4]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec[j + 4]));
		index += 2;
	}
#else
	// Modify deltas between hidden and output layers
	// For all nodes in hidden layer and bias neuron
	for (int j = 0; j <= NUMHIDDEN; j++) // Checked - because I rewrite it now it should be zero errorGradientOutput
	{
		int index = j * NUMOUTPUT;
		__mVec hiddenNeuronsVec = _mm_mul_ps(learningRateVec, _mm_set_ps1(hiddenNeurons[j]));
		int lastIndex = index + NUMOUTPUT;
		__mVec lastVec = _mm_loadu_ps(&deltaHiddenOutput[lastIndex]);
		for (int i = 0; i < simdNumOutput/VEC_LENGTH; i++)
		{
			// Calculate change in weight
			__mVec deltaHiddenOutputVec = _mm_loadu_ps(&deltaHiddenOutput[index + (i * VEC_LENGTH)]); // Very fast because in cache following loops. Although, penalty for not alignment.
			deltaHiddenOutputVec = _mm_add_ps(_mm_mul_ps(hiddenNeuronsVec, errorGradientsOutputVec[i]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec));
			_mVec_storeu_ps(&deltaHiddenOutput[index + (i * VEC_LENGTH)], deltaHiddenOutputVec); // Access not aligned - use storeu
		}
		// Fix overlap
		_mVec_storeu_ps(&deltaHiddenOutput[lastIndex], lastVec);
	}
#endif

#ifdef VERIFY
	for (int i = 0; i < 1510; i++)
		if (fabs(deltaHiddenOutput[i] - deltaHiddenOutputTrue[i]) > 0.00001)
			printf("1. !!!! ERROR !!!! (%d: %lf - %lf) %lf\n", i, deltaHiddenOutput[i], deltaHiddenOutputTrue[i], fabs(deltaHiddenOutput[i] - deltaHiddenOutputTrue[i]));


	float errorGradientsHiddenUnmodified[simdNumHidden];
	memcpy(&errorGradientsHiddenUnmodified, &errorGradientsHidden, simdNumHidden * 4);
	float deltaInputHiddenUnmodified[simdNumWeightsIH];
	memcpy(&deltaInputHiddenUnmodified, &deltaInputHidden, simdNumWeightsIH * 4);
	// modify deltas between input and hidden layers
	for (int i = 0; i <= NUMHIDDEN; i++)
	{
		// get error gradient for every hidden node
		errorGradientsHidden[i] = GetHiddenErrorGradient(i);
		// for all nodes in input layer and bias neuron
		for (int j = 0; j <= INPUTSIZE; j++)
		{
			const int weightIdx = GetInputHiddenWeightIndex(j, i);
			// calculate change in weight 
			deltaInputHidden[weightIdx] = LEARNINGRATE * inputNeurons[j] * errorGradientsHidden[i] + MOMENTUM * deltaInputHidden[weightIdx];
		}
	}

	float errorGradientsHiddenTrue[simdNumHidden];
	memcpy(&errorGradientsHiddenTrue, &errorGradientsHidden, simdNumHidden * 4);
	float deltaInputHiddenTrue[simdNumWeightsIH];
	memcpy(&deltaInputHiddenTrue, &deltaInputHidden, simdNumWeightsIH * 4);

	memcpy(&errorGradientsHidden, &errorGradientsHiddenUnmodified, simdNumHidden * 4);
	memcpy(&deltaInputHidden, &deltaInputHiddenUnmodified, simdNumWeightsIH * 4);
#endif


	union { int weightIndex_[VEC_LENGTH]; __mVeci weightIndexVec; };
	const __mVeci numoutputVec = _mm_set1_epi32(NUMOUTPUT);
	const __mVeci offsetVec = _mVec_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	// Precalculate hidden neurons error gradient
	for (int i = 0; i < simdNumHidden/VEC_LENGTH; i++) 
	{
		// Get error gradient for every hidden node
		__mVec weightedSumVec = _mm_setzero_ps();
		__mVeci iVec = _mVec_mullo_epi32(numoutputVec, _mVec_add_epi32(_mm_set1_epi32(i * VEC_LENGTH), offsetVec));
		for (int j = 0; j < NUMOUTPUT; j++)
		{
			// Calculate all indexes
			weightIndexVec = _mVec_add_epi32(iVec, _mm_set1_epi32(j));
			// Weighted sum costly to get all weights from hidden outputs (scattered) ...
			weightedSumVec = _mm_add_ps(weightedSumVec, _mm_mul_ps(
				_mVec_setr_ps(weightsHiddenOutput[weightIndex_[0]], weightsHiddenOutput[weightIndex_[1]], weightsHiddenOutput[weightIndex_[2]], weightsHiddenOutput[weightIndex_[3]], 
							  weightsHiddenOutput[weightIndex_[4]], weightsHiddenOutput[weightIndex_[5]], weightsHiddenOutput[weightIndex_[6]], weightsHiddenOutput[weightIndex_[7]]),
				_mm_set_ps1(errorGradientsOutput[j])));

		} 
		// Although it's improvement because we are accessing error gradients vec sequentially.
		errorGradientsHiddenVec[i] = _mm_mul_ps(hiddenNeuronsVec[i], _mm_mul_ps(_mm_sub_ps(onesVec, hiddenNeuronsVec[i]), weightedSumVec));
	}


	//union { __declspec(align(ALIGNMENT)) float deltaInputHidden_[simdNumHidden]; __mVec deltaInputHiddenLocalVec[simdNumHidden/VEC_LENGTH]; };

	// Modify deltas between input and hidden layers
	for (int j = 0; j <= INPUTSIZE; j++)
	{
		const int index = j * (NUMHIDDEN + 1);
		__mVec inputNeuronsVec = _mm_mul_ps(learningRateVec, _mm_set_ps1(inputNeurons[j]));
		const int lastIndex = index + (NUMHIDDEN + 1);
		__mVec lastVector = _mm_loadu_ps(&deltaInputHidden[lastIndex]);	// In case of 151 nodes and 152 one float would suffice, this is general solution, although slower
		//float fix = deltaInputHidden[lastIndex]; 
		//memcpy(&deltaInputHidden_, &deltaInputHidden[index], simdNumHidden*4); // It is strange but work most efficiently 
		// Modify deltas between input and hidden layers
		for (int i = 0; i < (simdNumHidden / VEC_LENGTH); i++)
		{
			// We have 3 channels to so maximum three SIMD operations per cycle... To optimize
			__mVec deltaInputHiddenLocalVec = _mm_mul_ps(momentumVec, _mm_loadu_ps(&deltaInputHidden[index + i * VEC_LENGTH])); // Very fast because in cache next loops. -> penalty for not alignment
			// Because we have latency of 1 (wow really good Intel) cycle it will calculate _mm_mul_ps(inputNeuronsVec,
			// errorGradientsHiddenVec[i] first then use deltaInputHiddenVec and get a final result
			// It will be efficient enough.
			// Calculate change in weight 
			//inputNeuronsVec = _mm_mul_ps(inputNeuronsVec, errorGradientsHiddenVec[i]);
			//deltaInputHidden_[i*VEC_LENGTH] *= MOMENTUM;
			//deltaInputHidden_[i*VEC_LENGTH+1] *= MOMENTUM;
			//__mVec temp = _mm_mul_ps(inputNeuronsVec, errorGradientsHiddenVec[i]);
			//deltaInputHidden_[i*VEC_LENGTH+2] *= MOMENTUM;
			//deltaInputHidden_[i*VEC_LENGTH+3] *= MOMENTUM;
			//_mm_mul_ps(momentumVec,
			__mVec result = _mm_add_ps(_mm_mul_ps(inputNeuronsVec, errorGradientsHiddenVec[i]), deltaInputHiddenLocalVec);
			_mVec_storeu_ps(&deltaInputHidden[index + i * VEC_LENGTH], result); // Access not aligned - use store
		}
		//deltaInputHidden[lastIndex] = fix;
		//memcpy(&deltaInputHidden[index], &deltaInputHidden_, (simdNumHidden-1) * 4);
		// Fix overlap
		_mVec_storeu_ps(&deltaInputHidden[lastIndex], lastVector);
	}

#ifdef VERIFY

	for (int i = 0; i < 151; i++)
		if (fabs(errorGradientsHidden[i] - errorGradientsHiddenTrue[i]) > 0.00001)
			printf("2. !!!! ERROR !!!! (%d: %lf - %lf)\n", i, errorGradientsHidden[i], errorGradientsHiddenTrue[i]);
	for (int i = 0; i < simdNumWeightsIH; i++)
		if (fabs(deltaInputHidden[i] - deltaInputHiddenTrue[i]) > 0.00001)
			printf("3. !!!! ERROR !!!! (%d: %lf - %lf) %lf\n", i, deltaInputHidden[i], deltaInputHiddenTrue[i], fabs(deltaInputHidden[i] - deltaInputHiddenTrue[i]));

	memcpy(&deltaInputHidden, &deltaInputHiddenTrue, simdNumWeightsIH * 4);
	memcpy(&errorGradientsHidden, &errorGradientsHiddenTrue, simdNumHidden * 4);
	memcpy(&deltaHiddenOutput, &deltaHiddenOutputTrue, 1520 * 4);
#endif
	//union { __declspec(align(ALIGNMENT)) float inputNeurons_[simdNumInput]; __mVec inputNeuronsLocalVec[simdNumInput/VEC_LENGTH]; };

	/*for (int i = 0; i < (simdNumHidden / VEC_LENGTH); i++)
	{
		for (int j = 0; j <= INPUTSIZE; j++)
		{
			//if (i == 0)
			//{
			//	inputNeuronsLocalVec[j>>2] = _mm_mul_ps(learningRateVec, _mm_set_ps1(inputNeurons[j]));
			//}
			const int index = j * (NUMHIDDEN + 1);
			__mVec deltaInputHiddenLocalVec = _mm_mul_ps(momentumVec, _mm_loadu_ps(&deltaInputHidden[index + i * VEC_LENGTH]));
			__mVec result = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(learningRateVec, _mm_set_ps1(inputNeurons[j])), errorGradientsHiddenVec[i]), deltaInputHiddenLocalVec);
			_mVec_storeu_ps(&deltaInputHidden[index + i * VEC_LENGTH], result); // Access not aligned - use store
		}
	}*/


	//Reverse loop Modify deltas between input and hidden layers
	/*for (int i = 0; i <= NUMHIDDEN; i++)
	{
		// get error gradient for every hidden node
		//errorGradientsHidden[i] = GetHiddenErrorGradient(i);
		// for all nodes in input layer and bias neuron

		union { float deltaInputHidden_[VEC_LENGTH]; __mVeci deltaInputHiddenVec; };
		for (int j = 0; j < simdNumInput/VEC_LENGTH; j++)
		{
			//const int weightIdx = GetInputHiddenWeightIndex(j, i);
			__mVec inputNeuronsVec = _mm_mul_ps(learningRateVec,_mVec_setr_ps(inputNeurons[j*VEC_LENGTH], inputNeurons[j*VEC_LENGTH+1], inputNeurons[j*VEC_LENGTH+2], inputNeurons[j*VEC_LENGTH+3],
																			   inputNeurons[j*VEC_LENGTH+4], inputNeurons[j*VEC_LENGTH + 5], inputNeurons[j*VEC_LENGTH + 6], inputNeurons[j*VEC_LENGTH + 7]));
			//return j * (NUMHIDDEN + 1) + i;
			// calculate change in weight 
			__mVec deltaInputHiddenVec = _mVec_setr_ps(deltaInputHidden[j * (NUMHIDDEN + 1) + i], deltaInputHidden[(j + 1) * (NUMHIDDEN + 1) + i], deltaInputHidden[(j + 2) * (NUMHIDDEN + 1) + i], deltaInputHidden[(j + 3) * (NUMHIDDEN + 1) + i],
													   deltaInputHidden[(j + 4) * (NUMHIDDEN + 1) + i], deltaInputHidden[(j + 5) * (NUMHIDDEN + 1) + i], deltaInputHidden[(j + 6) * (NUMHIDDEN + 1) + i], deltaInputHidden[(j + 7) * (NUMHIDDEN + 1) + i]);
			deltaInputHiddenVec = _mm_add_ps(_mm_mul_ps(inputNeuronsVec, errorGradientsHiddenVec[i]), _mm_mul_ps(momentumVec, deltaInputHiddenVec));
			for (int k = 0; j < VEC_LENGTH; k++)
				deltaInputHidden[(j + k) * (NUMHIDDEN + 1) + i] = deltaInputHidden_[VEC_LENGTH - k];
		}
	}*/

#else
	// modify deltas between hidden and output layers
	for (int i = 0; i < NUMOUTPUT; i++)
	{
		// get error gradient for every output node
		errorGradientsOutput[i] = GetOutputErrorGradient((float)expectedOutputs[i], outputNeurons[i]);
		// for all nodes in hidden layer and bias neuron
		for (int j = 0; j <= NUMHIDDEN; j++)
		{
			const int weightIdx = GetHiddenOutputWeightIndex(j, i);
			// calculate change in weight
			deltaHiddenOutput[weightIdx] = LEARNINGRATE * hiddenNeurons[j] * errorGradientsOutput[i] + MOMENTUM * deltaHiddenOutput[weightIdx];
		}
	}
	// modify deltas between input and hidden layers
	for (int i = 0; i <= NUMHIDDEN; i++)
	{
		// get error gradient for every hidden node
		errorGradientsHidden[i] = GetHiddenErrorGradient(i);
		// for all nodes in input layer and bias neuron
		for (int j = 0; j <= INPUTSIZE; j++)
		{
			const int weightIdx = GetInputHiddenWeightIndex(j, i);
			// calculate change in weight 
			deltaInputHidden[weightIdx] = LEARNINGRATE * inputNeurons[j] * errorGradientsHidden[i] + MOMENTUM * deltaInputHidden[weightIdx];
		}
	}
#endif
	// update the weights
	UpdateWeights();
}

const int* Network::Evaluate( const float* input )
{
	// Set input values
	memcpy(inputNeurons, input, INPUTSIZE * sizeof(float));
#ifdef VERIFY

	float *hiddenNeuronsUnmodified = new float[simdNumHidden];
	float *hiddenNeuronsTrue = new float[simdNumHidden];
	float *outputNeuronsTrue = new float[simdNumOutput];
	float *outputNeuronsUnmodified = new float[simdNumOutput];
	int *clampedOutputsTrue = new int[simdNumOutput];
	memcpy(hiddenNeuronsUnmodified, hiddenNeurons, simdNumHidden * 4);
	memcpy(outputNeuronsUnmodified, outputNeurons, simdNumOutput * 4);
	// update hidden neurons
	for (int i = 0; i < NUMHIDDEN; i++)
	{
		hiddenNeurons[i] = 0;
		// get weighted sum of pattern and bias neuron
		for (int j = 0; j <= INPUTSIZE; j++)
		{
			const int weightIdx = GetInputHiddenWeightIndex(j, i);
			hiddenNeurons[i] += inputNeurons[j] * weightsInputHidden[weightIdx];
		}
		// apply activation function
		hiddenNeurons[i] = SigmoidActivationFunction(hiddenNeurons[i]);
	}

	// calculate output values - include bias neuron
	for (int i = 0; i < NUMOUTPUT; i++)
	{
		outputNeurons[i] = 0;
		// get weighted sum of pattern and bias neuron
		for (int j = 0; j <= NUMHIDDEN; j++)
		{
			const int weightIdx = GetHiddenOutputWeightIndex(j, i);
			outputNeurons[i] += hiddenNeurons[j] * weightsHiddenOutput[weightIdx];
		}
		// apply activation function and clamp the result
		outputNeurons[i] = SigmoidActivationFunction(outputNeurons[i]);
		clampedOutputs[i] = ClampOutputValue(outputNeurons[i]);
	}

	memcpy(hiddenNeuronsTrue, hiddenNeurons, simdNumHidden * 4);
	memcpy(hiddenNeurons, hiddenNeuronsUnmodified, simdNumHidden * 4);
	memcpy(outputNeuronsTrue, outputNeurons, simdNumOutput * 4);
	memcpy(outputNeurons, outputNeuronsUnmodified, simdNumOutput * 4);
	memcpy(clampedOutputsTrue, clampedOutputs, simdNumOutput * 4);
#endif
	
#if SIMD & VECTORIZE_EVALUATE
	__mVec onesVec = _mm_set_ps1(1.0f);
	__mVec minusOnesVec = _mm_set_ps1(-1.0f);

	union { __declspec(align(ALIGNMENT)) float hn_[VEC_LENGTH]; __mVec hiddenNeuronsVec; };
	
	float hiddenNeuronFix = hiddenNeurons[NUMHIDDEN]; // To fix overwritten hidden neuron
	for (int i = 0; i < simdNumHidden / VEC_LENGTH; i++)
	{
		hiddenNeuronsVec = _mm_setzero_ps();
		const int index = i * VEC_LENGTH;
		// Get weighted sum of pattern and bias neuron
		for (int j = 0; j <= INPUTSIZE; j++)
		{
			int weightIdx = j * (NUMHIDDEN + 1) + index;
			__mVec inputNeuronsVec = _mm_set_ps1(inputNeurons[j]);
			__mVec weightsInputHiddenVec = _mm_loadu_ps(&weightsInputHidden[weightIdx]); 
			hiddenNeuronsVec = _mm_add_ps(hiddenNeuronsVec, _mm_mul_ps(inputNeuronsVec, weightsInputHiddenVec));
		}
		hiddenNeuronsVec = _mm_mul_ps(minusOnesVec, hiddenNeuronsVec);
		// Sadly no support/intrinsics to vectorize exp by SSE or AVX - could use extern libs 
		hiddenNeuronsVec = _mm_div_ps(onesVec, _mm_add_ps(onesVec, _mVec_setr_ps(expf(hn_[0]), expf(hn_[1]), expf(hn_[2]), expf(hn_[3]),
																				expf(hn_[4]), expf(hn_[5]), expf(hn_[6]), expf(hn_[7]))));
		_mm_store_ps(&hiddenNeurons[index], hiddenNeuronsVec); // Store it - it's aligned
	}
	hiddenNeurons[NUMHIDDEN] = hiddenNeuronFix; // To fix overwritten hidden neuron

	union { float on_[VEC_LENGTH]; __mVec outputNeuronsVec; };

	// Because we are vectorizing NUMOUTPUT consequently using AVX brings at most 1.5 speed up. 
	// Calculate output values - include bias neuron 
	for (int i = 0; i < simdNumOutput/VEC_LENGTH; i++)
	{
		outputNeuronsVec = _mm_setzero_ps(); 
		const int index = i * VEC_LENGTH;
		// Get weighted sum of pattern and bias neuron
		for (int j = 0; j <= NUMHIDDEN; j++)
		{
			int weightIdx = j * NUMOUTPUT + index;
			__mVec weightsHiddenOutputVec = _mm_loadu_ps(&weightsHiddenOutput[weightIdx]);
			__mVec hiddenNeuronsVec = _mm_set_ps1(hiddenNeurons[j]);
			outputNeuronsVec = _mm_add_ps(outputNeuronsVec, _mm_mul_ps(hiddenNeuronsVec, weightsHiddenOutputVec));
		}
		__mVec clampedOutputsVec = _mm_set_ps1(-1.0f);
		outputNeuronsVec = _mm_mul_ps(clampedOutputsVec, outputNeuronsVec);
		// Vectorize sigmoid activation function
		// Sadly no support/intrinsics to vectorize exp by SSE or AVX - could use extern libs 
		outputNeuronsVec = _mm_div_ps(onesVec, _mm_add_ps(onesVec, _mVec_setr_ps(expf(on_[0]), expf(on_[1]), expf(on_[2]), expf(on_[3]),
																				expf(on_[4]), expf(on_[5]), expf(on_[6]), expf(on_[7]))));
		// Apply activation function and clamp the result
		clampedOutputsVec = _mm_andnot_ps(_mm_cmp_ps(outputNeuronsVec, _mm_set_ps1(0.1f), _CMP_LT_OQ), clampedOutputsVec); // Get vec of -1 and 0 // _mm_cmp_ps works only with SSE 4.1 and higher
		clampedOutputsVec = _mm_add_ps(clampedOutputsVec, _mm_and_ps(_mm_set_ps1(2.0f), _mm_cmp_ps(outputNeuronsVec, _mm_set_ps1(0.9f), _CMP_GT_OQ))); // Add +2 to greater than 0.9

		_mm_store_ps(&outputNeurons[index], outputNeuronsVec); // It is aligned - store sufficient
		_mm_store_si128((__mVeci *)&clampedOutputs[index], _mm_cvtps_epi32(clampedOutputsVec)); // It is aligned but conversion is costly

	}
#ifdef VERIFY
	for (int i = 0; i < NUMHIDDEN+1; i++)
		if (fabs(hiddenNeuronsTrue[i] - hiddenNeurons[i]) > 0.00001)
			printf("1. !!!! ERROR EVAL !!!! (%d: %lf - %lf)\n", i, hiddenNeuronsTrue[i], hiddenNeurons[i]);
	for (int i = 0; i < NUMOUTPUT; i++)
		if (fabs(outputNeuronsTrue[i] - outputNeurons[i]) > 0.00001)
			printf("2. !!!! ERROR EVAL !!!! (%d: %lf - %lf) %lf\n", i, outputNeuronsTrue[i], outputNeurons[i], fabs(outputNeuronsTrue[i] - outputNeurons[i]));
	for (int i = 0; i < NUMOUTPUT; i++)
		if (clampedOutputsTrue[i] != clampedOutputs[i])
			printf("3. !!!! ERROR EVAL !!!! (%d: %d - %d)\n", i, clampedOutputs[i], clampedOutputsTrue[i]);

	memcpy(hiddenNeurons, hiddenNeuronsTrue, simdNumHidden * 4);
	memcpy(outputNeurons, outputNeuronsTrue, simdNumOutput * 4);
	memcpy(clampedOutputs, clampedOutputsTrue, simdNumOutput * 4);
#endif
#else
	// update hidden neurons
	for (int i = 0; i < NUMHIDDEN; i++)
	{
		hiddenNeurons[i] = 0;
		// get weighted sum of pattern and bias neuron
		for (int j = 0; j <= INPUTSIZE; j++)
		{
			const int weightIdx = GetInputHiddenWeightIndex(j, i);
			hiddenNeurons[i] += inputNeurons[j] * weightsInputHidden[weightIdx];
		}
		// apply activation function
		hiddenNeurons[i] = SigmoidActivationFunction(hiddenNeurons[i]);
	}
	// calculate output values - include bias neuron
	for (int i = 0; i < NUMOUTPUT; i++)
	{
		outputNeurons[i] = 0;
		// get weighted sum of pattern and bias neuron
		for (int j = 0; j <= NUMHIDDEN; j++)
		{
			const int weightIdx = GetHiddenOutputWeightIndex(j, i);
			outputNeurons[i] += hiddenNeurons[j] * weightsHiddenOutput[weightIdx];
		}
		// apply activation function and clamp the result
		outputNeurons[i] = SigmoidActivationFunction(outputNeurons[i]);
		clampedOutputs[i] = ClampOutputValue(outputNeurons[i]);
	}
#endif
	return clampedOutputs;
}

void Network::UpdateWeights()
{
#ifdef VERIFY
	float *weightsInputHiddenTrue = new float[simdNumWeightsIH];
	float *weightsHiddenOutputTrue = new float[simdNumHidden * NUMOUTPUT];
	float *weightsInputHiddenUnmodified = new float[simdNumWeightsIH];
	float *weightsHiddenOutputUnmodified = new float[(simdNumHidden * NUMOUTPUT)];

	memcpy(weightsInputHiddenUnmodified, weightsInputHidden, simdNumWeightsIH * sizeof(float));
	memcpy(weightsHiddenOutputUnmodified, weightsHiddenOutput, (simdNumHidden * NUMOUTPUT) * sizeof(float));
	// Input -> hidden weights
	for (int i = 0; i <= INPUTSIZE; i++) for (int j = 0; j <= NUMHIDDEN; j++)
	{
		const int weightIdx = GetInputHiddenWeightIndex(i, j);
		weightsInputHidden[weightIdx] += deltaInputHidden[weightIdx];
	}
	// Hidden -> output weights
	for (int i = 0; i <= NUMHIDDEN; i++) for (int j = 0; j < NUMOUTPUT; j++)
	{
		const int weightIdx = GetHiddenOutputWeightIndex(i, j);
		weightsHiddenOutput[weightIdx] += deltaHiddenOutput[weightIdx];
	}
	memcpy(weightsInputHiddenTrue, weightsInputHidden, simdNumWeightsIH * 4);
	memcpy(weightsHiddenOutputTrue, weightsHiddenOutput, (simdNumHidden * NUMOUTPUT) * 4);
	memcpy(weightsInputHidden, weightsInputHiddenUnmodified, simdNumWeightsIH * 4);
	memcpy(weightsHiddenOutput, weightsHiddenOutputUnmodified, (simdNumHidden * NUMOUTPUT) * 4);
#endif
#if SIMD & VECTORIZE_UPDATE_WEIGHTS
	// Input -> hidden weights
	for (int i = 0; i < simdNumWeightsIH / VEC_LENGTH ; i++)
	{
		weightsInputHiddenVec[i] = _mm_add_ps(weightsInputHiddenVec[i], deltaInputHiddenVec[i]);
	}
	// Hidden -> output weights
	for (int i = 0; i < (simdNumHidden * NUMOUTPUT) / VEC_LENGTH; i++)
	{
		weightsHiddenOutputVec[i] = _mm_add_ps(weightsHiddenOutputVec[i], deltaHiddenOutputVec[i]);
	}
#else
	// Input -> hidden weights
	for (int i = 0; i <= INPUTSIZE; i++) for (int j = 0; j <= NUMHIDDEN; j++)
	{
		const int weightIdx = GetInputHiddenWeightIndex(i, j);
		weightsInputHidden[weightIdx] += deltaInputHidden[weightIdx];
	}
	// Hidden -> output weights
	for (int i = 0; i <= NUMHIDDEN; i++) for (int j = 0; j < NUMOUTPUT; j++)
	{
		const int weightIdx = GetHiddenOutputWeightIndex(i, j);
		weightsHiddenOutput[weightIdx] += deltaHiddenOutput[weightIdx];
	}
#endif
#ifdef VERIFY

	// Input -> hidden weights
	for (int i = 0; i < simdNumWeightsIH; i++)
		if (fabs(weightsInputHidden[i] - weightsInputHiddenTrue[i]) > 0.00001)
			printf("1. !!!! ERROR !!!! (%d: %lf - %lf)\n", i, weightsInputHidden[i], weightsInputHiddenTrue[i]);
	// Hidden -> output weights
	for (int i = 0; i < (simdNumHidden * NUMOUTPUT); i++)
		if (fabs(weightsHiddenOutput[i] - weightsHiddenOutputTrue[i]) > 0.00001)
			printf("2. !!!! ERROR !!!! (%d: %lf - %lf)\n", i, weightsHiddenOutput[i], weightsHiddenOutputTrue[i]);

	memcpy(weightsInputHidden, weightsInputHiddenTrue, simdNumWeightsIH * 4);
	memcpy(weightsHiddenOutput, weightsHiddenOutputTrue, (simdNumHidden * NUMOUTPUT) * 4);
	delete[] weightsInputHiddenTrue;
	delete[] weightsHiddenOutputTrue;
	delete[] weightsInputHiddenUnmodified;
	delete[] weightsHiddenOutputUnmodified;
#endif
}

void Network::GetSetAccuracyAndMSE( const TrainingSet& set, float& accuracy, float& MSE ) 
{
	accuracy = 0, MSE = 0;
	float numIncorrectResults = 0;
	for( int i = 0; i < set.size; i++ )
	{
		const TrainingEntry& entry = set.entry[i];
		Evaluate( entry.inputs );
		// check if the network outputs match the expected outputs
		int correctResults = 0;
		for( int j = 0; j < NUMOUTPUT; j++ )
		{
			correctResults += (clampedOutputs[j] == entry.expected[j]);
			const float delta = outputNeurons[j] - entry.expected[j];
			MSE += delta * delta;
		}
		if (correctResults != NUMOUTPUT) numIncorrectResults++;
	}
	accuracy = 100.0f - (numIncorrectResults / set.size * 100.0f);
	MSE = MSE / (NUMOUTPUT * set.size);
}

} // namespace Tmpl8