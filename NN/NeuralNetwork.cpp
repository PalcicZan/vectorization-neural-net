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
	memset( hiddenNeurons, 0, NUMHIDDEN * 4 );
	memset( outputNeurons, 0, NUMOUTPUT * 4 );
	memset( clampedOutputs, 0, NUMOUTPUT * 4 );
	// set bias values
	inputNeurons[INPUTSIZE] = hiddenNeurons[NUMHIDDEN] = -1.0f;
	// create storage and initialize and layer weights
#if SIMD == OFF
	weightsInputHidden = new float[totalNumInputs * totalNumHiddens];
	weightsHiddenOutput = new float[totalNumHiddens * NUMOUTPUT];
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
	for (int i = 0; i < NUMOUTPUT; i++)
	{
		errorGradientsOutput[i] = GetOutputErrorGradient((float)expectedOutputs[i], outputNeurons[i]);
	}
	memcpy(&errorGradientsOutput[NUMOUTPUT], &errorGradientsOutput[0], NUMOUTPUT * sizeof(float));
	// get error gradient for every output node
	uint index = 0;
	const __mVec learningRateVec = _mm_set1_ps(LEARNINGRATE);
	const __mVec momentumVec = _mm_set1_ps(MOMENTUM);
	// 20 on each turn (SIMD_NUMHIDDEN * NUMOUTPUT) 
	for (int j = 0; j < (SIMD_NUMHIDDEN * NUMOUTPUT) / VEC_LENGTH; j += 5)
	{
		__mVec hd4 = _mm_mul_ps(learningRateVec, _mm_set_ps1(hiddenNeurons[index]));
		__mVec hd42 = _mVec_set_ps(hiddenNeurons[index], hiddenNeurons[index], hiddenNeurons[index + 1], hiddenNeurons[index + 1],0,0,0,0);
		__mVec hd41 = _mm_mul_ps(learningRateVec, _mm_set_ps1(hiddenNeurons[index + 1]));
		deltaHiddenOutputVec[j] = _mm_add_ps(_mm_mul_ps(hd4, errorGradientsOutputVec[0]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec[j]));
		deltaHiddenOutputVec[j + 1] = _mm_add_ps(_mm_mul_ps(hd4, errorGradientsOutputVec[1]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec[j + 1]));
		deltaHiddenOutputVec[j + 2] = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(learningRateVec, hd42), errorGradientsOutputVec[2]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec[j + 2]));
		deltaHiddenOutputVec[j + 3] = _mm_add_ps(_mm_mul_ps(hd41, errorGradientsOutputVec[3]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec[j + 3]));
		deltaHiddenOutputVec[j + 4] = _mm_add_ps(_mm_mul_ps(hd41, errorGradientsOutputVec[4]), _mm_mul_ps(momentumVec, deltaHiddenOutputVec[j + 4]));
		index += 2;
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
	//const __mVec learningRateVec = _mm_set1_ps(LEARNINGRATE);
	//const __mVec momentumVec = _mm_set1_ps(MOMENTUM);
	//union { int weightIndex_[VEC_LENGTH]; __mVeci weightIndexVec; };
	//const __mVeci vecLengthVec = _mm_set1_epi32(VEC_LENGTH);
	//const __mVeci numoutputVec = _mm_set1_epi32(NUMOUTPUT);
	//const __mVeci offsetVec = _mm_set_epi32(0,1,2,3);// 4, 5, 6, 7);
	__mVec onesVec = _mm_set1_ps(1.f);
	for (int i = 0; i <= SIMD_NUMHIDDEN / VEC_LENGTH; i++)
	{
		// get error gradient for every hidden node
		__mVec weightedSum4 = _mm_setzero_ps();
		const int offset = i * VEC_LENGTH;
		//__mVeci iVec = _mm_mul_epi32(numoutputVec, _mm_add_epi32(_mm_mul_epi32(_mm_set1_epi32(i), vecLengthVec), offsetVec));
		for (int j = 0; j < NUMOUTPUT; j++)
		{
			//weightIndexVec = _mm_add_epi32(iVec, _mm_set1_epi32(j));
			//(offset + 4)* NUMOUTPUT + j
			// Get hidden output weight index
			const int weightIdx1 = GetHiddenOutputWeightIndex(offset, j);
			const int weightIdx2 = GetHiddenOutputWeightIndex(offset + 1, j);
			const int weightIdx3 = GetHiddenOutputWeightIndex(offset + 2, j);
			const int weightIdx4 = GetHiddenOutputWeightIndex(offset + 3, j);

			weightedSum4 = _mm_add_ps(weightedSum4, _mm_mul_ps(
				_mVec_set_ps(weightsHiddenOutput[weightIdx1], weightsHiddenOutput[weightIdx2], weightsHiddenOutput[weightIdx3], weightsHiddenOutput[weightIdx4]
					,weightsHiddenOutput[(offset + 4 )* NUMOUTPUT + j], weightsHiddenOutput[(offset + 5)* NUMOUTPUT + j],
					weightsHiddenOutput[(offset + 6)* NUMOUTPUT + j], weightsHiddenOutput[(offset + 7)* NUMOUTPUT + j]), _mm_set_ps1(errorGradientsOutput[j])));
				/*_mVec_set_ps(weightsHiddenOutput[weightIndex_[0]], weightsHiddenOutput[weightIndex_[1]], weightsHiddenOutput[weightIndex_[2]], weightsHiddenOutput[weightIndex_[3]]
			,weightsHiddenOutput[weightIndex_[4]], weightsHiddenOutput[weightIndex_[5]],
			weightsHiddenOutput[weightIndex_[6]], weightsHiddenOutput[weightIndex_[7]]*/

		}
		errorGradientsHiddenVec[i] = _mm_mul_ps(hiddenNeuronsVec[i], _mm_mul_ps(_mm_sub_ps(onesVec, hiddenNeuronsVec[i]), weightedSum4));
	}
	//for (int i = 0; i <= NUMHIDDEN; i++) errorGradientsHidden[i] = GetHiddenErrorGradient(i);
	
	// modify deltas between input and hidden layers
	// for all nodes in input layer and bias neuron
	for (int j = 0; j <= INPUTSIZE; j++)
	{
		const int index = j * (NUMHIDDEN + 1);
		__mVec inputNeuronsVec = _mm_mul_ps(learningRateVec, _mm_set_ps1(inputNeurons[j]));
		// modify deltas between input and hidden layers
		for (int i = 0; i <= SIMD_NUMHIDDEN / VEC_LENGTH; i++)
		{
			__mVec deltaInputHiddenVec = _mm_loadu_ps(&deltaInputHidden[index + i * VEC_LENGTH]);
			//int dihIndex = index + i * 4;
			//__mVec dih4 = _mVec_set_ps(deltaInputHidden[dihIndex], deltaInputHidden[dihIndex+1], deltaInputHidden[dihIndex+2], deltaInputHidden[dihIndex+3]);
			__mVec errorGradientsHiddenVec = _mm_load_ps(&errorGradientsHidden[i * VEC_LENGTH]);
			// calculate change in weight 
			deltaInputHiddenVec = _mm_add_ps(_mm_mul_ps(inputNeuronsVec, errorGradientsHiddenVec), _mm_mul_ps(momentumVec, deltaInputHiddenVec));
			_mm_storeu_ps(&deltaInputHidden[index + i * VEC_LENGTH], deltaInputHiddenVec);
		}
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
	// update hidden neurons
#if SIMD & VECTORIZE_EVALUATE
	__mVec onesVec = _mm_set_ps1(1.0f);
	union { float hn_[VEC_LENGTH]; __mVec hiddenNeuronsVec; };
	for (int i = 0; i < SIMD_NUMHIDDEN / VEC_LENGTH; i++)
	{
		//hiddenNeurons[i] = 0;
		hiddenNeuronsVec = _mm_setzero_ps();
		int index = i * VEC_LENGTH;
		// get weighted sum of pattern and bias neuron
		for (int j = 0; j <= INPUTSIZE; j++)
		{
			int weightIdx = j * (NUMHIDDEN + 1) + index;
			
			__mVec inputNeuronsVec = _mm_set_ps1(inputNeurons[j]);
			// Cache miss - fallowed by hits
			//__mVec weightsInputHiddenVec = _mVec_set_ps(weightsInputHidden[weightIdx], weightsInputHidden[weightIdx + 1], weightsInputHidden[weightIdx + 2], weightsInputHidden[weightIdx + 3],
			//											weightsInputHidden[weightIdx + 4], weightsInputHidden[weightIdx + 5], weightsInputHidden[weightIdx + 6], weightsInputHidden[weightIdx + 7]); // AVX else ignored!
			__mVec weightsInputHiddenVec = _mm_loadu_ps(&weightsInputHidden[weightIdx]);
			hiddenNeuronsVec = _mm_add_ps(hiddenNeuronsVec, _mm_mul_ps(inputNeuronsVec, weightsInputHiddenVec));
		}
		hiddenNeuronsVec = _mm_mul_ps(_mm_set_ps1(-1.0f), hiddenNeuronsVec);
		// Sadly no support/intrinsics to vectorize expf by Intel - could use 
		hiddenNeuronsVec = _mm_div_ps(onesVec, _mm_add_ps(onesVec, _mVec_set_ps(expf(hn_[3]), expf(hn_[2]), expf(hn_[1]), expf(hn_[0]),
																				expf(hn_[4]), expf(hn_[5]), expf(hn_[6]), expf(hn_[7]))));
		_mm_store_ps(&hiddenNeurons[index], hiddenNeuronsVec); // It should be aligned
	}

	union { float on_[VEC_LENGTH]; __mVec outputNeuronsVec; };
	// because we vectorize NUMOUTPUT if using AVX brings at most 1.5 speed up. 
	// calculate output values - include bias neuron 
	for (int i = 0; i < ((NUMOUTPUT + VEC_LENGTH) / VEC_LENGTH); i++) //(NUMOUTPUT+2) / VEC_LENGTH
	{
		outputNeuronsVec = _mm_setzero_ps(); 
		int index = i * VEC_LENGTH;
		// get weighted sum of pattern and bias neuron
		for (int j = 0; j <= NUMHIDDEN; j++)
		{
			const int weightIdx = j * NUMOUTPUT + index;
			__mVec weightsHiddenOutputVec = _mVec_set_ps(weightsHiddenOutput[weightIdx], weightsHiddenOutput[weightIdx + 1], weightsHiddenOutput[weightIdx + 2], weightsHiddenOutput[weightIdx + 3],
														weightsHiddenOutput[weightIdx + 4], weightsHiddenOutput[weightIdx + 5], weightsHiddenOutput[weightIdx + 6], weightsHiddenOutput[weightIdx + 7]); // AVX else ignored!
			__mVec hiddenNeuronsVec = _mm_set_ps1(hiddenNeurons[j]);
			outputNeuronsVec = _mm_add_ps(outputNeuronsVec, _mm_mul_ps(hiddenNeuronsVec, weightsHiddenOutputVec));
		}
		__mVec clampedOutputsVec = _mm_set_ps1(-1.0f);
		outputNeuronsVec = _mm_mul_ps(clampedOutputsVec, outputNeuronsVec);
		// Vectorize sigmoid activation function
		// Sadly no support/intrinsics to vectorize expf by Intel - could use 
		outputNeuronsVec = _mm_div_ps(onesVec, _mm_add_ps(onesVec, _mVec_set_ps(expf(on_[0]), expf(on_[1]), expf(on_[2]), expf(on_[3]),
																				expf(on_[4]), expf(on_[5]), expf(on_[6]), expf(on_[7]))));
		// apply activation function and clamp the result
		clampedOutputsVec = _mm_andnot_ps(_mm_cmplt_ps(outputNeuronsVec, _mm_set_ps1(0.1f) // SET ALL TO ZERO TODO!!!!
#if SIMD & AVX
		, _CMP_LE_OQ
#endif
		), clampedOutputsVec); // Get vec of -1 and 0
		clampedOutputsVec = _mm_add_ps(clampedOutputsVec, _mm_and_ps(_mm_set_ps1(2.0f), _mm_cmpgt_ps(outputNeuronsVec, _mm_set_ps1(0.9f)
#if SIMD & AVX
		, _CMP_GT_OQ
#endif
		))); // Add +2 to greater than 0.9

		_mm_storeu_ps(&outputNeurons[index], outputNeuronsVec); // It is aligned - store sufficient
		_mm_storeu_si128((__mVeci *)&clampedOutputs[index], _mm_cvttps_epi32(clampedOutputsVec));// It is aligned - store sufficient
	}
#else
	for(int i = 0; i < NUMHIDDEN; i++)
	{
	hiddenNeurons[i] = 0;
	// get weighted sum of pattern and bias neuron
	for (int j = 0; j <= INPUTSIZE; j++)
	{
		const int weightIdx = GetInputHiddenWeightIndex(j, i);
		//int GetInputHiddenWeightIndex( int inputIdx, int hiddenIdx ) const { return inputIdx * (NUMHIDDEN + 1) + hiddenIdx; }
		hiddenNeurons[i] += inputNeurons[j] * weightsInputHidden[weightIdx];

	}
	// apply activation function
	hiddenNeurons[i] = SigmoidActivationFunction(hiddenNeurons[i]);//return 1.0f / (1.0f + expf(-x));
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
	for (int i = 0; i < (SIMD_NUMHIDDEN * NUMOUTPUT) / VEC_LENGTH; i++)
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