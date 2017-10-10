//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// Based on Bobby Anguelov's code
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include "..\precomp.h"

namespace Tmpl8 {

Network::Network()
{
	// initialize neural net
	InitializeNetwork();
	InitializeWeights();
	// initialize trainer (calloc: malloc + clear to zero)
#if SIMD != OFF
	deltaInputHidden[(INPUTSIZE + 1) * (NUMHIDDEN + 1) + 1] = { 0.f };
	deltaHiddenOutput[SIMD_NUMHIDDEN * NUMOUTPUT] = { 0.f };
	errorGradientsHidden[SIMD_NUMHIDDEN] = { 0.f };
	errorGradientsOutput[SIMD_NUMOUTPUT] = { 0.f };
#else
	deltaInputHidden = (float*)calloc((INPUTSIZE + 1) * (NUMHIDDEN + 1), sizeof(float));
	deltaHiddenOutput = (float*)calloc((NUMHIDDEN + 1) * NUMOUTPUT, sizeof(float));
	errorGradientsHidden = (float*)calloc(NUMHIDDEN + 1, sizeof(float));
	errorGradientsOutput = (float*)calloc(NUMOUTPUT, sizeof(float));
#endif
}

void Network::InitializeNetwork()
{
	// create storage and initialize the neurons and the outputs
	// add bias neurons
	const int totalNumInputs = INPUTSIZE + 1, totalNumHiddens = NUMHIDDEN + 1;
	memset( inputNeurons, 0, INPUTSIZE * 4 );
	memset( hiddenNeurons, 0, SIMD_NUMHIDDEN * 4 );
	memset( outputNeurons, 0, SIMD_NUMOUTPUT * 4 );
	memset( clampedOutputs, 0, SIMD_NUMOUTPUT * 4 );
	// set bias values
	inputNeurons[INPUTSIZE] = hiddenNeurons[NUMHIDDEN] = -1.0f;
	// create storage and initialize and layer weights
#if SIMD == OFF
	weightsInputHidden = new float[totalNumInputs * totalNumHiddens];
	weightsHiddenOutput = new float[totalNumHiddens * NUMOUTPUT];
#else
	weightsHiddenOutput[SIMD_NUMHIDDEN * NUMOUTPUT] = { 0.0f };
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
	//printf("%d,%d,%d, %d\n", simdNumHidden, simdNumInput, simdNumOutput, test);
	// Probably not to vectorize
	for( int i = 0; i < set.size; i++ )
	{
		const TrainingEntry& entry = set.entry[i];
		// feed inputs through network and back propagate errors

		//Vectorized evaluate
		Evaluate( entry.inputs );
		//Vectorized backpropagate
		Backpropagate( entry.expected );

		// check all outputs from neural network against desired values
		bool resultCorrect = true;
		for( int j = 0; j < NUMOUTPUT; j++ )
		{
			if (clampedOutputs[j] != entry.expected[j]) resultCorrect = false;
			const float delta = outputNeurons[j] - entry.expected[j];
			MSE += delta * delta;
		}
		if (!resultCorrect) incorrectEntries++;
	}
	// update training accuracy and MSE
	trainingSetAccuracy = 100.0f - (incorrectEntries / set.size * 100.0f);
	trainingSetMSE = MSE / (NUMOUTPUT * set.size);
}

void Network::Backpropagate(const int* expectedOutputs)
{
#if SIMD & VECTORIZE_BACKPROPAGATE
	// modify deltas between hidden and output layers

	for (int i = 0; i < NUMOUTPUT; i++) // Too few to bother
	{
		errorGradientsOutput[i] = GetOutputErrorGradient((float)expectedOutputs[i], outputNeurons[i]);
	}
	memcpy(&errorGradientsOutput[NUMOUTPUT], &errorGradientsOutput[0],10 * sizeof(float));
	const __mVec learningRateVec = _mm_set1_ps(LEARNINGRATE);
	const __mVec momentumVec = _mm_set1_ps(MOMENTUM);
#if SIMD & 0
	// Extend error gradient output to repeat itself special case for SSE example 
	memcpy(&errorGradientsOutput[NUMOUTPUT], &errorGradientsOutput[0], NUMOUTPUT * sizeof(float));
	// get error gradient for every output node
	int index = 0;
	// 20 on each turn (SIMD_NUMHIDDEN * NUMOUTPUT) 
	for (int j = 0; j < (SIMD_NUMHIDDEN * NUMOUTPUT) / VEC_LENGTH; j += 5) // Optimized to length of 4
	{
		__mVec hd4 = _mm_mul_ps(learningRateVec, _mm_set_ps1(hiddenNeurons[index]));
		__mVec hd42 = _mVec_setr_ps(hiddenNeurons[index + 1], hiddenNeurons[index + 1], hiddenNeurons[index], hiddenNeurons[index],0,0,0,0);
		__mVec hd41 = _mm_mul_ps(learningRateVec, _mm_set_ps1(hiddenNeurons[index + 1]));
		deltaHiddenOutputVec[j] = _mm_add_ps(_mm_mul_ps(hd4, errorGradientsOutputVec[0]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec[j]));
		deltaHiddenOutputVec[j + 1] = _mm_add_ps(_mm_mul_ps(hd4, errorGradientsOutputVec[1]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec[j + 1]));
		// Edge case
		deltaHiddenOutputVec[j + 2] = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(learningRateVec, hd42), errorGradientsOutputVec[2]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec[j + 2]));
		deltaHiddenOutputVec[j + 3] = _mm_add_ps(_mm_mul_ps(hd41, errorGradientsOutputVec[3]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec[j + 3]));
		deltaHiddenOutputVec[j + 4] = _mm_add_ps(_mm_mul_ps(hd41, errorGradientsOutputVec[4]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec[j + 4]));
		index += 2;
	}
#else
	// for all nodes in hidden layer and bias neuron
	for (int j = 0; j <= NUMHIDDEN; j++)
	{
		int index = j * NUMOUTPUT;
		__mVec hiddenNeuronsVec = _mm_mul_ps(learningRateVec, _mm_set_ps1(hiddenNeurons[j]));
		__mVec lastVec = _mm_loadu_ps(&deltaHiddenOutput[index + ((simdNumOutput/VEC_LENGTH)* VEC_LENGTH) - (simdNumOutput - 10)]);
		for (int i = 0; i < simdNumOutput/VEC_LENGTH; i++)
		{
			// calculate change in weight
			//deltaHiddenOutput[weightIdx] = LEARNINGRATE * hiddenNeurons[j] * errorGradientsOutput[i] + MOMENTUM * deltaHiddenOutput[weightIdx];
			__mVec deltaHiddenOutputVec = _mm_loadu_ps(&deltaHiddenOutput[index + (i * VEC_LENGTH)]);
			//__mVec errorGradientsOutputVec = _mm_load_ps(&errorGradientsOutput[i * VEC_LENGTH]);
			deltaHiddenOutputVec = _mm_add_ps(_mm_mul_ps(hiddenNeuronsVec, errorGradientsOutputVec[i]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec));
			_mVec_storeu_ps(&deltaHiddenOutput[index + (i * VEC_LENGTH)], deltaHiddenOutputVec); // Access not aligned - use storeu
			//_mm256_storeu2_m128(&deltaHiddenOutput[index + (i * VEC_LENGTH)+4], &deltaHiddenOutput[index + (i * VEC_LENGTH)], deltaHiddenOutputVec);
		}
		_mVec_storeu_ps(&deltaHiddenOutput[index + ((simdNumOutput / VEC_LENGTH)* VEC_LENGTH) - (simdNumOutput - 10)], lastVec); // Access not aligned - use storeu
	}

	// modify deltas between hidden and output layers
	/*for (int i = 0; i < NUMOUTPUT; i++)
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
	}*/
#endif
	/*errorGradientsOutput[10] = errorGradientsOutput[2];
	errorGradientsOutput[11] = errorGradientsOutput[3];
	errorGradientsOutput[12] = errorGradientsOutput[4];
	errorGradientsOutput[13] = errorGradientsOutput[5];
	errorGradientsOutput[14] = errorGradientsOutput[6];
	errorGradientsOutput[15] = errorGradientsOutput[7];*/
	//memcpy(&weightsHiddenOutput[1510], &weightsHiddenOutput[0], NUMOUTPUT * sizeof(float));
	//memcpy(&deltaHiddenOutput[1510], &deltaHiddenOutput[0], NUMOUTPUT * sizeof(float));
	//memcpy(&errorGradientsOutput[NUMOUTPUT], &errorGradientsOutput[0], NUMOUTPUT * sizeof(float));

	//hiddenNeurons[151] = hiddenNeurons[0];
	union { int weightIndex_[VEC_LENGTH]; __mVeci weightIndexVec; };
	const __mVeci numoutputVec = _mm_set1_epi32(NUMOUTPUT);
	const __mVeci offsetVec = _mVec_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	const __mVec onesVec = _mm_set1_ps(1.f);
	// Precalculate hidden error gradient
	/*for (int i = 0; i <= simdNumHidden/VEC_LENGTH; i++)
	{
		// get error gradient for every hidden node
		__mVec weightedSumVec = _mm_setzero_ps();
		//__mVeci iVec = _mm_mullo_epi32(numoutputVec, _mm_add_epi32(_mm_mullo_epi32(_mm_set1_epi32(i), vecLengthVec), offsetVec));
		__mVeci iVec = _mVec_mullo_epi32(numoutputVec, _mVec_add_epi32(_mm_set1_epi32(i * VEC_LENGTH), offsetVec));
		for (int j = 0; j < NUMOUTPUT; j++)
		{
			weightIndexVec = _mVec_add_epi32(iVec, _mm_set1_epi32(j));
			// Get hidden output weight index
			//return hiddenIdx * NUMOUTPUT + outputIdx;
			//const int weightIdx1 = GetHiddenOutputWeightIndex(offset, j);
			//const int weightIdx2 = GetHiddenOutputWeightIndex(offset + 1, j);
			//const int weightIdx3 = GetHiddenOutputWeightIndex(offset + 2, j);
			//const int weightIdx4 = GetHiddenOutputWeightIndex(offset + 3, j);
			//printf("%d %d %d %d\n", weightIndex_[0], weightIndex_[1], weightIndex_[2], weightIndex_[3]);
			weightedSumVec = _mm_add_ps(weightedSumVec, _mm_mul_ps(
				_mVec_setr_ps(weightsHiddenOutput[weightIndex_[0]], weightsHiddenOutput[weightIndex_[1]], weightsHiddenOutput[weightIndex_[2]], weightsHiddenOutput[weightIndex_[3]], 
							  weightsHiddenOutput[weightIndex_[4]], weightsHiddenOutput[weightIndex_[5]], weightsHiddenOutput[weightIndex_[6]], weightsHiddenOutput[weightIndex_[7]]),
				_mm_set_ps1(errorGradientsOutput[j])));

		}
		errorGradientsHiddenVec[i] = _mm_mul_ps(hiddenNeuronsVec[i], _mm_mul_ps(_mm_sub_ps(onesVec, hiddenNeuronsVec[i]), weightedSumVec));
		// NOT REWRITEN SHOULD HAVE CORRECT DATA!!!!!!!!!!!!!!
		//_mm_store_ps(&errorGradientsHidden[i * VEC_LENGTH], _mm_mul_ps(hiddenNeuronsVec[i], _mm_mul_ps(_mm_sub_ps(onesVec, hiddenNeuronsVec[i]), weightedSumVec))); // Access not aligned - use storeu
	}*/
	for (int i = 0; i <= NUMHIDDEN; i++) errorGradientsHidden[i] = GetHiddenErrorGradient(i);

	//float deltaInputHiddenTemp[118536];
	//memcpy(&deltaInputHiddenTemp, &deltaInputHidden, 118536*4);

	// modify deltas between input and hidden layers
	/*for (int i = 0; i <= NUMHIDDEN; i++)
	{
		// for all nodes in input layer and bias neuron
		for (int j = 0; j <= INPUTSIZE; j++)
		{
			const int weightIdx = GetInputHiddenWeightIndex(j, i);
			// calculate change in weight 
			//deltaInputHidden[weightIdx] = LEARNINGRATE * inputNeurons[j] * errorGradientsHidden[i] + MOMENTUM * deltaInputHidden[weightIdx];
			deltaInputHidden[weightIdx] = LEARNINGRATE * inputNeurons[j] * errorGradientsHidden[i] + MOMENTUM * deltaInputHidden[weightIdx];
		}
	}*/
	// weightIndex 22952, i = 0; j = 152; inputNeuron = 0.000; 	eghi[0] = 4.78672446e-0; dih[22952] == deltaInputHidden[weightIdx]	1.25043176e-09	float

	//errorGradientsHidden[151] = errorGradientsHidden[0];
	//memcpy(&errorGradientsHiddenVec[NUMOUTPUT], &errorGradientsOutput, NUMOUTPUT * sizeof(float));

	for (int j = 0; j <= INPUTSIZE; j++) // CHECKED !!!
	{
		const int index = j * (NUMHIDDEN + 1);
		__mVec inputNeuronsVec = _mm_mul_ps(learningRateVec, _mm_set_ps1(inputNeurons[j]));
		// modify deltas between input and hidden layers
		__mVec lastVector = _mm_loadu_ps(&deltaInputHidden[index + ((simdNumHidden / VEC_LENGTH)* VEC_LENGTH) - (simdNumHidden - 151)]); //- offset
		for (int i = 0; i < (simdNumHidden / VEC_LENGTH); i++)
		{
			__mVec deltaInputHiddenVec = _mm_loadu_ps(&deltaInputHidden[index + i * VEC_LENGTH]);
			//__mVec deltaInputHiddenCalculationVec = _mm_loadu_ps(&deltaInputHiddenTemp[index + (i ) * VEC_LENGTH*((bool)(i*1))]);
			//__mVec dih4 = _mVec_setr_ps(deltaInputHidden[dihIndex], deltaInputHidden[dihIndex+1], deltaInputHidden[dihIndex+2], deltaInputHidden[dihIndex+3]);
			//__mVec errorGradientsHiddenVec = _mm_load_ps(&errorGradientsHidden[i * VEC_LENGTH]);
			// Calculate change in weight 
			deltaInputHiddenVec = _mm_add_ps(_mm_mul_ps(inputNeuronsVec, errorGradientsHiddenVec[i]), _mm_mul_ps(momentumVec, deltaInputHiddenVec));
			_mVec_storeu_ps(&deltaInputHidden[index + i * VEC_LENGTH], deltaInputHiddenVec); // Access not aligned - use storeu
			//float a = deltaInputHidden[index + i * VEC_LENGTH];
			//float b = deltaInputHiddenTemp[index + i * VEC_LENGTH];
			//printf("%lf : %lf (%u)\n", a, b, a == b);
			//if (a != b)
			//	printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%d\n", i);
		}
		// Repair offset
		_mVec_storeu_ps(&deltaInputHidden[index + ((simdNumHidden / VEC_LENGTH)* VEC_LENGTH) - (simdNumHidden - 151)], lastVector);
	}

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
	// set input values
	memcpy( inputNeurons, input, INPUTSIZE * sizeof( float ) );

#if SIMD & VECTORIZE_EVALUATE
	__mVec onesVec = _mm_set_ps1(1.0f);
	union { float hn_[VEC_LENGTH]; __mVec hiddenNeuronsVec; };
	for (int i = 0; i < SIMD_NUMHIDDEN / VEC_LENGTH; i++)
	{
		//hiddenNeurons[i] = 0;
		hiddenNeuronsVec = _mm_setzero_ps();
		const int index = i * VEC_LENGTH;
		// get weighted sum of pattern and bias neuron
		for (int j = 0; j <= INPUTSIZE; j++)
		{
			int weightIdx = j * (NUMHIDDEN + 1) + index;
			__mVec inputNeuronsVec = _mm_set_ps1(inputNeurons[j]);
			// Cache miss - fallowed by hits
			//__mVec weightsInputHiddenVec = _mVec_setr_ps(weightsInputHidden[weightIdx], weightsInputHidden[weightIdx + 1], weightsInputHidden[weightIdx + 2], weightsInputHidden[weightIdx + 3],
			//											weightsInputHidden[weightIdx + 4], weightsInputHidden[weightIdx + 5], weightsInputHidden[weightIdx + 6], weightsInputHidden[weightIdx + 7]); // AVX else ignored!
			__mVec weightsInputHiddenVec = _mm_loadu_ps(&weightsInputHidden[weightIdx]);
			hiddenNeuronsVec = _mm_add_ps(hiddenNeuronsVec, _mm_mul_ps(inputNeuronsVec, weightsInputHiddenVec));
		}
		hiddenNeuronsVec = _mm_mul_ps(_mm_set_ps1(-1.0f), hiddenNeuronsVec);
		// Sadly no support/intrinsics to vectorize expf by Intel - could use 
		hiddenNeuronsVec = _mm_div_ps(onesVec, _mm_add_ps(onesVec, _mVec_setr_ps(expf(hn_[0]), expf(hn_[1]), expf(hn_[2]), expf(hn_[3]),
																				expf(hn_[4]), expf(hn_[5]), expf(hn_[6]), expf(hn_[7]))));
		_mm_store_ps(&hiddenNeurons[index], hiddenNeuronsVec); // Store it - it's aligned
	}

	union { float on_[VEC_LENGTH]; __mVec outputNeuronsVec; };
	// Because we are vectorizing NUMOUTPUT consequently using AVX brings at most 1.5 speed up. 
	// Calculate output values - include bias neuron 
	for (int i = 0; i < ((NUMOUTPUT + VEC_LENGTH) / VEC_LENGTH); i++) //(NUMOUTPUT+2) / VEC_LENGTH
	{
		outputNeuronsVec = _mm_setzero_ps(); 
		const int index = i * VEC_LENGTH;
		// get weighted sum of pattern and bias neuron
		for (int j = 0; j <= NUMHIDDEN; j++)
		{
			int weightIdx = j * NUMOUTPUT + index;
			//__mVec weightsHiddenOutputVec = _mVec_setr_ps(weightsHiddenOutput[weightIdx], weightsHiddenOutput[weightIdx + 1], weightsHiddenOutput[weightIdx + 2], weightsHiddenOutput[weightIdx + 3],
			//											weightsHiddenOutput[weightIdx + 4], weightsHiddenOutput[weightIdx + 5], weightsHiddenOutput[weightIdx + 6], weightsHiddenOutput[weightIdx + 7]); // AVX else ignored!
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
		clampedOutputsVec = _mm_andnot_ps(_mm_cmp_ps(outputNeuronsVec, _mm_set_ps1(0.1f), _CMP_LE_OQ), clampedOutputsVec); // Get vec of -1 and 0 // _mm_cmp_ps works only with SSE 4.1 and higher
		clampedOutputsVec = _mm_add_ps(clampedOutputsVec, _mm_and_ps(_mm_set_ps1(2.0f), _mm_cmp_ps(outputNeuronsVec, _mm_set_ps1(0.9f), _CMP_GT_OQ))); // Add +2 to greater than 0.9

		_mm_store_ps(&outputNeurons[index], outputNeuronsVec); // It is aligned - store sufficient
		_mm_store_si128((__mVeci *)&clampedOutputs[index], _mm_cvttps_epi32(clampedOutputsVec)); // It is aligned - store sufficient
	}
#else	
	// update hidden neurons
	for(int i = 0; i < NUMHIDDEN; i++)
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
#if SIMD & VECTORIZE_UPDATE_WEIGHTS
	// input -> hidden weights
	for (int i = 0; i < ((INPUTSIZE + 1) * (NUMHIDDEN + 1) + 1) / VEC_LENGTH ; i++)
	{
		weightsInputHiddenVec[i] = _mm_add_ps(weightsInputHiddenVec[i], deltaInputHiddenVec[i]);
	}
	// hidden -> output weights
	for (int i = 0; i < (simdNumHidden * NUMOUTPUT) / VEC_LENGTH; i++)
	{
		weightsHiddenOutputVec[i] = _mm_add_ps(weightsHiddenOutputVec[i], deltaHiddenOutputVec[i]);
	}
#else
	// input -> hidden weights
	for (int i = 0; i <= INPUTSIZE; i++) for (int j = 0; j <= NUMHIDDEN; j++)
	{
		const int weightIdx = GetInputHiddenWeightIndex(i, j);
		weightsInputHidden[weightIdx] += deltaInputHidden[weightIdx];
	}

	// hidden -> output weights
	for( int i = 0; i <= NUMHIDDEN; i++ ) for ( int j = 0; j < NUMOUTPUT; j++ )
	{
		const int weightIdx = GetHiddenOutputWeightIndex( i, j );
		weightsHiddenOutput[weightIdx] += deltaHiddenOutput[weightIdx];
	}
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