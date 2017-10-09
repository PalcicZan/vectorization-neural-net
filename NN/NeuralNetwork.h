//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------
// A simple neural network supporting only a single hidden layer
// Do not change defines
#define OFF 0
#define SSE 0b1000
#define AVX 0b10000

#define VECTORIZE_BACKPROPAGATE 0b1
#define VECTORIZE_EVALUATE 0b10
#define VECTORIZE_UPDATE_WEIGHTS 0b100
#define VECTORIZE_ALL (VECTORIZE_BACKPROPAGATE | VECTORIZE_EVALUATE | VECTORIZE_UPDATE_WEIGHTS)

// Free to change defines
#define REFSPEED 4000.0
// Default is that everything is vectorized (SSE | VECTORIZE_ALL)
// Otherwise you had to specify all VECTORIZE_* you want to use 
// (e.g. SIMD (SSE | VECTORIZE_BACKPROPAGATE | VECTORIZE_EVALUATE))
#define SIMD (SSE | VECTORIZE_ALL)

#if SIMD & SSE
#define VEC_LENGTH 4
#define ALIGNMENT 16
typedef __m128 __mVec;
typedef __m128i __mVeci;
#define SIMD_INPUTSIZE (INPUTSIZE)
#define SIMD_NUMOUTPUT 20
#define SIMD_NUMHIDDEN (NUMHIDDEN+2)
inline static __mVec _mVec_set_ps(float a, float b, float c, float d, float e, float f, float g, float h) { return _mm_set_ps(a, b, c, d); };

#elif SIMD & AVX

#define VEC_LENGTH 8
#define ALIGNMENT 32
typedef __m256 __mVec;
typedef __m256i __mVeci;
#define SIMD_INPUTSIZE (INPUTSIZE)
#define SIMD_NUMOUTPUT 20
#define SIMD_NUMHIDDEN (NUMHIDDEN+2)
#define _mm_setzero_ps _mm256_setzero_ps
#define _mm_mul_ps _mm256_mul_ps
#define _mm_div_ps _mm256_div_ps
#define _mm_add_ps _mm256_add_ps
#define _mm_sub_ps _mm256_sub_ps
#define _mm_cmplt_ps _mm256_cmp_ps
#define _mm_cmpgt_ps _mm256_cmp_ps
#define _mm_andnot_ps _mm256_andnot_ps
#define _mm_and_ps _mm256_and_ps
#define _mm_load_ps _mm256_load_ps
#define _mm_loadu_ps _mm256_loadu_ps
#define _mm_store_ps _mm256_store_ps
#define _mm_storeu_ps _mm256_storeu_ps
#define _mm_set_ps _mm256_set_ps
#define _mm_set_ps1 _mm256_set1_ps
// Integer operators
#define _mm_set_epi32 _mm256_set_epi32
#define _mm_set1_epi32 _mm256_set1_epi32
#define _mm_mul_epi32 _mm256_mul_epi32
#define _mm_add_epi32 _mm256_add_epi32
#define _mm_store_si128 _mm256_store_si256
#define _mm_storeu_si128 _mm256_storeu_si256
#define _mm_cvtps_epi32 _mm256_cvtps_epi32
#define _mm_cvttps_epi32 _mm256_cvttps_epi32
inline static __mVec _mVec_set_ps(float a, float b, float c, float d, float e, float f, float g, float h) { return _mm256_set_ps(a, b, c, d, e, f, g, h); };
#else
#define VEC_LENGTH 1
#endif
#if SIMD == SSE
/*inline static __mVec _mVec_mul_ps(__mVec a, __mVec b) { return _mm_mul_ps(a, b); };
inline static __mVec _mVec_add_ps(__mVec a, __mVec b) { return _mm_add_ps(a, b); };
inline static __mVec _mVec_set_ps(float a, float b, float c, float d) { return _mm_set_ps(a, b, c, d); };
inline static __mVec _mVec_set_ps1(float a) { return _mm_set_ps1(a); };
inline static __mVec _mVec_div_ps(__mVec a, __mVec b) { return _mm_div_ps(a, b); };
inline static __mVec _mVec_load_ps(float const *a) { return _mm_load_ps(a); };
inline static __mVec _mVec_loadu_ps(float const *a) { return _mm_loadu_ps(a); };
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
#if SIMD != OFF
	union { float __declspec(align(ALIGNMENT)) inputNeurons[INPUTSIZE + VEC_LENGTH]; __mVec inputNeuronsVec[(INPUTSIZE + VEC_LENGTH) / VEC_LENGTH]; };
	union { float __declspec(align(ALIGNMENT))  hiddenNeurons[SIMD_NUMHIDDEN]; __mVec hiddenNeuronsVec[SIMD_NUMHIDDEN / VEC_LENGTH]; };
#else
	float inputNeurons[INPUTSIZE + 1];
	float hiddenNeurons[NUMHIDDEN + 1];
#endif

public:
#if SIMD != OFF
	__declspec(align(ALIGNMENT)) float outputNeurons[SIMD_NUMOUTPUT];//float outputNeurons[NUMOUTPUT];
#else
	float outputNeurons[NUMOUTPUT];
#endif
private:
#if SIMD != OFF
	__declspec(align(16)) int clampedOutputs[SIMD_NUMOUTPUT];
	union {	float __declspec(align(ALIGNMENT)) weightsInputHidden[(INPUTSIZE + 1)*(NUMHIDDEN + 1) + 1]; __mVec weightsInputHiddenVec[((INPUTSIZE + 1) * (NUMHIDDEN + 1) + 1) / VEC_LENGTH];};
	union {	float __declspec(align(ALIGNMENT)) weightsHiddenOutput[SIMD_NUMHIDDEN * NUMOUTPUT]; __mVec weightsHiddenOutputVec[SIMD_NUMHIDDEN * NUMOUTPUT / VEC_LENGTH];};
	// training data	
	union { float __declspec(align(ALIGNMENT)) deltaInputHidden[(INPUTSIZE + 1) * (NUMHIDDEN + 1) + 1]; __mVec deltaInputHiddenVec[((INPUTSIZE + 1) * (NUMHIDDEN + 1) + 1) / VEC_LENGTH]; };
	union { float __declspec(align(ALIGNMENT)) deltaHiddenOutput[SIMD_NUMHIDDEN * NUMOUTPUT]; __mVec deltaHiddenOutputVec[(SIMD_NUMHIDDEN * NUMOUTPUT) / VEC_LENGTH]; };
	union { float __declspec(align(ALIGNMENT)) errorGradientsHidden[SIMD_NUMHIDDEN]; __mVec errorGradientsHiddenVec[SIMD_NUMHIDDEN / VEC_LENGTH]; };
	union { float __declspec(align(ALIGNMENT)) errorGradientsOutput[SIMD_NUMOUTPUT]; __mVec errorGradientsOutputVec[SIMD_NUMOUTPUT / VEC_LENGTH]; };
#else
	int clampedOutputs[NUMOUTPUT];
	float* weightsInputHidden;
	float* weightsHiddenOutput;
	// training data	
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