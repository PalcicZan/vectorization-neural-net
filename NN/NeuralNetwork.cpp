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
#if SIMD == AVX || SIMD == SSE
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

		// TODO: Vectorize
		Evaluate( entry.inputs );
		// TODO: Sync
		// TODO: Vectorize
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
#if SIMD == AVX || SIMD == SSE
	// modify deltas between hidden and output layers
	for (int i = 0; i < NUMOUTPUT; i++)
	{
		errorGradientsOutput[i] = GetOutputErrorGradient((float)expectedOutputs[i], outputNeurons[i]);
	}
	memcpy(&errorGradientsOutput[NUMOUTPUT], &errorGradientsOutput[0], NUMOUTPUT * sizeof(float));
	// get error gradient for every output node
	uint index = 0;
	const __m128 learningRate4 = _mm_set1_ps(LEARNINGRATE);
	const __m128 momentum4 = _mm_set1_ps(MOMENTUM);
	// 20 on each turn (SIMD_NUMHIDDEN * NUMOUTPUT) 
	for (int j = 0; j < (SIMD_NUMHIDDEN * NUMOUTPUT) / 4; j += 5)
	{
		__m128 hd4 = _mm_mul_ps(learningRate4, _mm_set_ps1(hiddenNeurons[index]));
		__m128 hd42 = _mm_set_ps(hiddenNeurons[index], hiddenNeurons[index], hiddenNeurons[index + 1], hiddenNeurons[index + 1]);
		__m128 hd41 = _mm_mul_ps(learningRate4, _mm_set_ps1(hiddenNeurons[index + 1]));
		deltaHiddenOutputVec[j] = _mm_add_ps(_mm_mul_ps(hd4, errorGradientsOutputVec[0]), _mm_mul_ps(momentum4, deltaHiddenOutputVec[j]));
		deltaHiddenOutputVec[j + 1] = _mm_add_ps(_mm_mul_ps(hd4, errorGradientsOutputVec[1]), _mm_mul_ps(momentum4, deltaHiddenOutputVec[j + 1]));
		deltaHiddenOutputVec[j + 2] = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(learningRate4, hd42), errorGradientsOutputVec[2]), _mm_mul_ps(momentum4, deltaHiddenOutputVec[j + 2]));
		deltaHiddenOutputVec[j + 3] = _mm_add_ps(_mm_mul_ps(hd41, errorGradientsOutputVec[3]), _mm_mul_ps(momentum4, deltaHiddenOutputVec[j + 3]));
		deltaHiddenOutputVec[j + 4] = _mm_add_ps(_mm_mul_ps(hd41, errorGradientsOutputVec[4]), _mm_mul_ps(momentum4, deltaHiddenOutputVec[j + 4]));
		index += 2;
	}

	__m128 ones4 = _mm_set1_ps(1.f);
	for (int i = 0; i <= SIMD_NUMHIDDEN / 4; i++)
	{
		// get error gradient for every hidden node
		__m128 weightedSum4 = _mm_setzero_ps();
		for (int j = 0; j < NUMOUTPUT; j++)
		{
			const int weightIdx1 = GetHiddenOutputWeightIndex(i * 4, j);
			const int weightIdx2 = GetHiddenOutputWeightIndex((i * 4) + 1, j);
			const int weightIdx3 = GetHiddenOutputWeightIndex((i * 4) + 2, j);
			const int weightIdx4 = GetHiddenOutputWeightIndex((i * 4) + 3, j);
			weightedSum4 = _mm_add_ps(weightedSum4, _mm_mul_ps(
				_mm_set_ps(weightsHiddenOutput[weightIdx1], weightsHiddenOutput[weightIdx2], weightsHiddenOutput[weightIdx3], weightsHiddenOutput[weightIdx4])
				, _mm_set_ps1(errorGradientsOutput[j])));
		}
		errorGradientsHiddenVec[i] = _mm_mul_ps(hiddenNeuronsVec[i], _mm_mul_ps(_mm_sub_ps(ones4, hiddenNeuronsVec[i]), weightedSum4));
	}
	//for (int i = 0; i <= NUMHIDDEN; i++) errorGradientsHidden[i] = GetHiddenErrorGradient(i);
	
	// modify deltas between input and hidden layers
	// for all nodes in input layer and bias neuron
	for (int j = 0; j <= INPUTSIZE; j++)
	{
		int index = j * (NUMHIDDEN + 1);
		__m128 in4 = _mm_mul_ps(learningRate4, _mm_set_ps1(inputNeurons[j]));
		// modify deltas between input and hidden layers
		for (int i = 0; i <= SIMD_NUMHIDDEN / 4; i++)
		{
			__m128 dih4 = _mm_loadu_ps(&deltaInputHidden[index + i * 4]);
			//int dihIndex = index + i * 4;
			//__m128 dih4 = _mm_set_ps(deltaInputHidden[dihIndex], deltaInputHidden[dihIndex+1], deltaInputHidden[dihIndex+2], deltaInputHidden[dihIndex+3]);
			__m128 egh4 = _mm_load_ps(&errorGradientsHidden[i * 4]);
			// calculate change in weight 
			dih4 = _mm_add_ps(_mm_mul_ps(in4, egh4), _mm_mul_ps(momentum4, dih4));
			_mm_storeu_ps(&deltaInputHidden[index + i * 4], dih4);
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
#if SIMD > OFF

	__m128 ones4 = _mm_set1_ps(1.0f);
	union { float hn_[4]; __m128 hn4; };
	for (int i = 0; i < SIMD_NUMHIDDEN/4; i++)
	{
		//hiddenNeurons[i] = 0;
		hn4 = _mm_setzero_ps();
		int index = i * 4;
		// get weighted sum of pattern and bias neuron
		for (int j = 0; j <= INPUTSIZE; j++)
		{
			int wihIndex = j * (NUMHIDDEN + 1) + index;
			
			__m128 in4 = _mm_set1_ps(inputNeurons[j]);
			__m128 wih4 = _mm_set_ps(weightsInputHidden[wihIndex],
									 weightsInputHidden[wihIndex + 1],
									 weightsInputHidden[wihIndex + 2],
									 weightsInputHidden[wihIndex + 3]);
			//__m128 wih4 = _mm_loadu_ps(&weightsInputHidden[wihIndex]);
			hn4 = _mm_add_ps(hn4, _mm_mul_ps(in4, wih4));
		}
		hn4 = _mm_mul_ps(_mm_set1_ps(-1.0f), hn4);
		hn4 = _mm_div_ps(ones4, _mm_add_ps(ones4, _mm_set_ps(expf(hn_[0]), expf(hn_[1]), expf(hn_[2]), expf(hn_[3]))));
		_mm_storeu_ps(&hiddenNeurons[index], hn4);
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
#endif
	// calculate output values - include bias neuron
	for( int i = 0; i < NUMOUTPUT; i++ )// TODO
	{
		outputNeurons[i] = 0;
		// get weighted sum of pattern and bias neuron
		for( int j = 0; j <= NUMHIDDEN; j++ )
		{
			const int weightIdx = GetHiddenOutputWeightIndex( j, i );
			outputNeurons[i] += hiddenNeurons[j] * weightsHiddenOutput[weightIdx];
		}
		// apply activation function and clamp the result
		outputNeurons[i] = SigmoidActivationFunction( outputNeurons[i] ); 
		clampedOutputs[i] = ClampOutputValue( outputNeurons[i] );
	}
	return clampedOutputs;
}

void Network::UpdateWeights()
{
#if SIMD == AVX || SIMD == SSE
	// input -> hidden weights
	for (int i = 0; i < ((INPUTSIZE + 1) * (NUMHIDDEN + 1) + 1) / 4 ; i++)
	{
		weightsInputHiddenVec[i] = _mm_add_ps(weightsInputHiddenVec[i], deltaInputHiddenVec[i]);
	}
	// hidden -> output weights
	for (int i = 0; i < (SIMD_NUMHIDDEN * NUMOUTPUT) / 4; i++)
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