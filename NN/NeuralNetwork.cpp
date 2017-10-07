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
#if SIMD > 0
	deltaInputHidden[(INPUTSIZE + 1) * (NUMHIDDEN + 1) + 1] = {};
	deltaHiddenOutput[SIMD_NUMHIDDEN * NUMOUTPUT] = {};
	errorGradientsHidden[SIMD_NUMHIDDEN] = {};
	errorGradientsOutput[SIMD_NUMOUTPUT] = {};
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
	weightsInputHidden = new float[totalNumInputs * totalNumHiddens];
	weightsHiddenOutput = new float[totalNumHiddens * NUMOUTPUT];
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
		const int* expectedOutputs = entry.expected;
		// TODO: Sync
		// TODO: Vectorize
		//Backpropagate( entry.expected );
#if SIMD == BACKPROPAGATE || SIMD == ALL 
		// modify deltas between hidden and output layers
		for (int i = 0; i < NUMOUTPUT; i++)
		{
			errorGradientsOutput[i] = GetOutputErrorGradient((float)expectedOutputs[i], outputNeurons[i]);
		}
		memcpy(&errorGradientsOutput[NUMOUTPUT], errorGradientsOutput, NUMOUTPUT * sizeof(float));
		/*for (int i = 0; i < NUMOUTPUT; i++)
		{
			errorGradientsOutput[i + NUMOUTPUT] = errorGradientsOutput[i];
		}*/
		// get error gradient for every output node
		//__m128 desiredValue4 = _mm_set_ps((float)expectedOutputs[i * 4], (float)expectedOutputs[(i * 4) + 1], (float)expectedOutputs[(i * 4) + 2], (float)expectedOutputs[(i * 4) + 3])
		//outputValue * (1.0f - outputValue) * (desiredValue - outputValue);
		//__m128 i4 = _mm_set_ps1(i);
		uint index = 0;
		const __m128 learningRate = _mm_set1_ps(LEARNINGRATE);
		const __m128 momentum = _mm_set1_ps(MOMENTUM);
		for (int j = 0; j < (SIMD_NUMHIDDEN * NUMOUTPUT) / 4; j += 5)
		{
			//const int weightIdx = GetHiddenOutputWeightIndex(j, i); // int GetHiddenOutputWeightIndex(int hiddenIdx, int outputIdx) const { return hiddenIdx * NUMOUTPUT + outputIdx; }
			//hiddenIdx * NUMOUTPUT + outputIdx;						// calculate change in weight
			//__m128 j4 = _mm_set_ps(j, j + 1, j + 2, j + 3);
			//j4 = _mm_fmadd_ps(j4, numOutput4, i4);
			__m128 hd4 = _mm_set_ps1(hiddenNeurons[index]);
			__m128 hd41 = _mm_set_ps1(hiddenNeurons[index + 1]);
			__m128 hd42 = _mm_set_ps(hiddenNeurons[index], hiddenNeurons[index], hiddenNeurons[index + 1], hiddenNeurons[index + 1]);
			hd4 = _mm_mul_ps(learningRate, hd4);
			deltaHiddenOutput4[j] = _mm_add_ps(_mm_mul_ps(hd4, errorGradientsOutput4[0]), _mm_mul_ps(momentum, deltaHiddenOutput4[j]));
			deltaHiddenOutput4[j + 1] = _mm_add_ps(_mm_mul_ps(hd4, errorGradientsOutput4[1]), _mm_mul_ps(momentum, deltaHiddenOutput4[j + 1]));
			//__m128 hd4 = _mm_set_ps(hiddenNeurons[index], hiddenNeurons[index], hiddenNeurons[index], hiddenNeurons[index]);
			deltaHiddenOutput4[j + 2] = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(learningRate, hd42), errorGradientsOutput4[2]), _mm_mul_ps(momentum, deltaHiddenOutput4[j + 2]));

			//hd4 = _mm_set_ps1(hiddenNeurons[index]);
			//__m128 hd4 = _mm_set_ps(hiddenNeurons[index], hiddenNeurons[index], hiddenNeurons[index], hiddenNeurons[index]);
			//deltaHiddenOutput4[j + 4] = _mm_fmadd_ps(_mm_mul_ps(learningRate, hd41), errorGradientsOutput4[4], _mm_mul_ps(momentum, deltaHiddenOutput4[j + 4]));
			hd41 = _mm_mul_ps(learningRate, hd41);
			deltaHiddenOutput4[j + 3] = _mm_add_ps(_mm_mul_ps(hd41, errorGradientsOutput4[3]), _mm_mul_ps(momentum, deltaHiddenOutput4[j + 3]));
			deltaHiddenOutput4[j + 4] = _mm_add_ps(_mm_mul_ps(hd41, errorGradientsOutput4[4]), _mm_mul_ps(momentum, deltaHiddenOutput4[j + 4]));
			index =+ 2;
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

		// check all outputs from neural network against desired values
		bool resultCorrect = true;
		// TODO: I should vectorize
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
	// modify deltas between hidden and output layers
#if SIMD == BACKPROPAGATE || SIMD == ALL
	//inline float GetOutputErrorGradient(float desiredValue, float outputValue) const { return outputValue * (1.0f - outputValue) * (desiredValue - outputValue); }
	for (int i = 0; i < NUMOUTPUT; i++)
	{
		errorGradientsOutput[i] = GetOutputErrorGradient((float)expectedOutputs[i], outputNeurons[i]);
	}		

	for (int i = 0; i < NUMOUTPUT; i++)
	{
		errorGradientsOutput[i+NUMOUTPUT] = errorGradientsOutput[i];
	}
	// get error gradient for every output node
	//__m128 desiredValue4 = _mm_set_ps((float)expectedOutputs[i * 4], (float)expectedOutputs[(i * 4) + 1], (float)expectedOutputs[(i * 4) + 2], (float)expectedOutputs[(i * 4) + 3])
	//outputValue * (1.0f - outputValue) * (desiredValue - outputValue);
	//__m128 i4 = _mm_set_ps1(i);
	uint index = 0;
	const __m128 learningRate = _mm_set1_ps(LEARNINGRATE);
	const __m128 momentum = _mm_set1_ps(MOMENTUM);
	for (int j = 0; j < (SIMD_NUMHIDDEN * NUMOUTPUT) / 4; j += 5)
	{
		//const int weightIdx = GetHiddenOutputWeightIndex(j, i); // int GetHiddenOutputWeightIndex(int hiddenIdx, int outputIdx) const { return hiddenIdx * NUMOUTPUT + outputIdx; }
																//hiddenIdx * NUMOUTPUT + outputIdx;						// calculate change in weight
		//__m128 j4 = _mm_set_ps(j, j + 1, j + 2, j + 3);
		//j4 = _mm_fmadd_ps(j4, numOutput4, i4);
		__m128 hd4 = _mm_set_ps1(hiddenNeurons[index]);
		__m128 hd41 = _mm_set_ps1(hiddenNeurons[index + 1]);
		__m128 hd42 = _mm_set_ps(hiddenNeurons[index], hiddenNeurons[index], hiddenNeurons[index + 1], hiddenNeurons[index + 1]);
		deltaHiddenOutput4[j] = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(learningRate, hd4),errorGradientsOutput4[0]),_mm_mul_ps(momentum, deltaHiddenOutput4[j]));
		deltaHiddenOutput4[j + 1] = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(learningRate, hd4), errorGradientsOutput4[1]), _mm_mul_ps(momentum, deltaHiddenOutput4[j + 1]));
		//__m128 hd4 = _mm_set_ps(hiddenNeurons[index], hiddenNeurons[index], hiddenNeurons[index], hiddenNeurons[index]);
		deltaHiddenOutput4[j+2] = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(learningRate, hd42), errorGradientsOutput4[2]), _mm_mul_ps(momentum, deltaHiddenOutput4[j+2]));

		//hd4 = _mm_set_ps1(hiddenNeurons[index]);
		//__m128 hd4 = _mm_set_ps(hiddenNeurons[index], hiddenNeurons[index], hiddenNeurons[index], hiddenNeurons[index]);
		//deltaHiddenOutput4[j + 4] = _mm_fmadd_ps(_mm_mul_ps(learningRate, hd41), errorGradientsOutput4[4], _mm_mul_ps(momentum, deltaHiddenOutput4[j + 4]));
		deltaHiddenOutput4[j+3] = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(learningRate, hd41), errorGradientsOutput4[3]), _mm_mul_ps(momentum, deltaHiddenOutput4[j+3]));
		deltaHiddenOutput4[j+4] = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(learningRate, hd41), errorGradientsOutput4[4]), _mm_mul_ps(momentum, deltaHiddenOutput4[j+4]));
		index =+ 2;
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
#endif
	// modify deltas between input and hidden layers
	for( int i = 0; i <= NUMHIDDEN; i++ )
	{
		// get error gradient for every hidden node
		errorGradientsHidden[i] = GetHiddenErrorGradient( i );
		// for all nodes in input layer and bias neuron
		for( int j = 0; j <= INPUTSIZE; j++ )
		{
			const int weightIdx = GetInputHiddenWeightIndex( j, i );
			// calculate change in weight 
			deltaInputHidden[weightIdx] = LEARNINGRATE * inputNeurons[j] * errorGradientsHidden[i] + MOMENTUM * deltaInputHidden[weightIdx];
		}
	}
	// update the weights
	UpdateWeights();
}

const int* Network::Evaluate( const float* input )
{
	// set input values
	memcpy( inputNeurons, input, INPUTSIZE * sizeof( float ) );
	// update hidden neurons
	for( int i = 0; i < NUMHIDDEN; i++ )
	{
		hiddenNeurons[i] = 0;
		// get weighted sum of pattern and bias neuron
		for( int j = 0; j <= INPUTSIZE; j++ )
		{
			const int weightIdx = GetInputHiddenWeightIndex( j, i );
			hiddenNeurons[i] += inputNeurons[j] * weightsInputHidden[weightIdx];
		}
		// apply activation function
		hiddenNeurons[i] = SigmoidActivationFunction( hiddenNeurons[i] );
	}
	// calculate output values - include bias neuron
	for( int i = 0; i < NUMOUTPUT; i++ )
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
	// input -> hidden weights
	for( int i = 0; i <= INPUTSIZE; i++ ) for( int j = 0; j <= NUMHIDDEN; j++ )
	{
		const int weightIdx = GetInputHiddenWeightIndex( i, j );
		weightsInputHidden[weightIdx] += deltaInputHidden[weightIdx];
	}
	// hidden -> output weights
	for( int i = 0; i <= NUMHIDDEN; i++ ) for ( int j = 0; j < NUMOUTPUT; j++ )
	{
		const int weightIdx = GetHiddenOutputWeightIndex( i, j );
		weightsHiddenOutput[weightIdx] += deltaHiddenOutput[weightIdx];
	}
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