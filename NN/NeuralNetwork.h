//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------
// A simple neural network supporting only a single hidden layer


#define OFF 1
#define SSE 4
#define AVX 8

#define REFSPEED 4000.0

#define SIMD SSE
#if SIMD == SSE
typedef __m128 __mVec;
#define SIMD_INPUTSIZE (INPUTSIZE)
#define SIMD_NUMOUTPUT 20
#define SIMD_NUMHIDDEN (NUMHIDDEN+2)
#elif SIMD == AVX
#define SIMD_NUMHIDDEN (NUMHIDDEN+2)
typedef __m256 __mVec;
#define _mm_mul_ps _mm256_mul_ps
#define _mm_add_ps _mm256_add_ps
#define _mm_set_ps _mm256_set_ps
#define _mm_set_ps1 _mm256_set_ps1
#define _mm_load_ps _mm256_load_ps
#define _mm_loadu_ps _mm256_loadu_ps
#define _mm_store_ps _mm256_store_ps
#define _mm_storeu_ps _mm256_storeu_ps
#endif
#if SIMD == SSE
/*inline static __mVec _mVec_mul_ps(__mVec a, __mVec b) { return _mm_mul_ps(a, b); };
inline static __mVec _mVec_add_ps(__mVec a, __mVec b) { return _mm_add_ps(a, b); };
inline static __mVec _mVec_set_ps(float a, float b, float c, float d) { return _mm_set_ps(a, b, c, d); };
inline static __mVec _mVec_set_ps1(float a) { return _mm_set_ps1(a); };
inline static __mVec _mVec_div_ps(__mVec a, __mVec b) { return _mm_div_ps(a, b); };
inline static __mVec _mVec_load_ps(float const*a) { return _mm_load_ps(a); };
inline static __mVec _mVec_loadu_ps(float const*a) { return _mm_loadu_ps(a); };
inline static void _mVec_storeu_ps(float *a, __m128 b) { _mm_storeu_ps(a, b); };
inline static void _mVec_store_ps(float *a, __m128 b) { return _mm_store_ps(a, b); };*/
#endif


namespace Tmpl8 {

struct TrainingEntry
{
	float inputs[INPUTSIZE];
	int expected[NUMOUTPUT];
};

struct TrainingSet
{
	TrainingEntry* entry;
	int size;
};

struct TrainingData
{
	TrainingData( int t, int g, int v )
	{
		trainingSet.entry = new TrainingEntry[t];
		trainingSet.size = t;
		generalizationSet.entry = new TrainingEntry[g];
		generalizationSet.size = g;
		validationSet.entry = new TrainingEntry[v];
		validationSet.size = v;
	}
	TrainingSet trainingSet;
	TrainingSet generalizationSet;
	TrainingSet validationSet;
};

class Network
{
	friend class NetworkTrainer;
	inline static float SigmoidActivationFunction( float x ) { return 1.0f / (1.0f + expf( -x )); }
	inline static int ClampOutputValue( float x ) { if ( x < 0.1f ) return 0; else if ( x > 0.9f ) return 1; else return -1; }
	inline float GetOutputErrorGradient( float desiredValue, float outputValue ) const { return outputValue * (1.0f - outputValue) * (desiredValue - outputValue); }
	int GetInputHiddenWeightIndex( int inputIdx, int hiddenIdx ) const { return inputIdx * (NUMHIDDEN + 1) + hiddenIdx; }
	int GetHiddenOutputWeightIndex( int hiddenIdx, int outputIdx ) const { return hiddenIdx * NUMOUTPUT + outputIdx; }
public:
	Network();
	const int* Evaluate( const float* input );
	void Train( const TrainingData& trainingData );
	const float* GetInputHiddenWeights() const { return weightsInputHidden; }
	const float* GetHiddenOutputWeights() const { return weightsHiddenOutput; }
	void LoadWeights( const float* weights );
	void SaveWeights( float* weights );
	void InitializeNetwork();
	void InitializeWeights();
	float GetHiddenErrorGradient( int hiddenIdx ) const;
	void RunEpoch( const TrainingSet& trainingSet );
	void Backpropagate( const int* expectedOutputs );
	void UpdateWeights();
	void GetSetAccuracyAndMSE( const TrainingSet& trainingSet, float& accuracy, float& mse );
private:
	// neural net data
#if SIMD > 1
	union { float __declspec(align(16)) inputNeurons[INPUTSIZE + 4]; __m128 inputNeuronsVec[(INPUTSIZE + 4) / 4]; };
	union { float __declspec(align(16))  hiddenNeurons[SIMD_NUMHIDDEN]; __m128 hiddenNeuronsVec[SIMD_NUMHIDDEN / 4]; };
#else
	float inputNeurons[INPUTSIZE + 1];
	float hiddenNeurons[NUMHIDDEN + 1];
#endif

public:
	float outputNeurons[NUMOUTPUT];
private:
	int clampedOutputs[NUMOUTPUT]; 

#if SIMD > 0
	union {	float __declspec(align(16)) weightsInputHidden[(INPUTSIZE + 1)*(NUMHIDDEN + 1) + 1]; __m128 weightsInputHiddenVec[((INPUTSIZE + 1) * (NUMHIDDEN + 1) + 1) / SIMD];};
	union {	float __declspec(align(16)) weightsHiddenOutput[SIMD_NUMHIDDEN * NUMOUTPUT]; __m128 weightsHiddenOutputVec[SIMD_NUMHIDDEN * NUMOUTPUT / SIMD];};
#else
	float* weightsInputHidden;
	float* weightsHiddenOutput;
#endif
	// training data		
#if SIMD > 0
	union { float __declspec(align(16)) deltaInputHidden[(INPUTSIZE + 1) * (NUMHIDDEN + 1) + 1]; __m128 deltaInputHiddenVec[((INPUTSIZE + 1) * (NUMHIDDEN + 1)+1) / 4]; };
	union { float __declspec(align(16)) deltaHiddenOutput[SIMD_NUMHIDDEN * NUMOUTPUT]; __m128 deltaHiddenOutputVec[(SIMD_NUMHIDDEN * NUMOUTPUT) / SIMD]; };
	union { float __declspec(align(16)) errorGradientsHidden[SIMD_NUMHIDDEN]; __m128 errorGradientsHiddenVec[SIMD_NUMHIDDEN / 4]; };
	union { float __declspec(align(16)) errorGradientsOutput[SIMD_NUMOUTPUT]; __m128 errorGradientsOutputVec[SIMD_NUMOUTPUT / 4]; };
#else
	float*	 deltaInputHidden;				// delta for input hidden layer
	float*   deltaHiddenOutput;				// delta for hidden output layer
	float*   errorGradientsHidden;			// error gradients for the hidden layer
	float*   errorGradientsOutput;			// error gradients for the outputs
#endif
	int      currentEpoch;					// epoch counter
	float    trainingSetAccuracy;
	float    validationSetAccuracy;
	float    generalizationSetAccuracy;
	float    trainingSetMSE;
	float    validationSetMSE;
	float    generalizationSetMSE;
};

} // namespace Tmpl8