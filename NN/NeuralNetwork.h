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

// Get closest higher number to "i" which is dividable by 2^div
inline static constexpr int getClosestDiv(int i, int div) { return ((i >> div) + (bool)(i & 0b111)) << div; };
//-------------------------------------------------------------------------
// Free to change defines
//-------------------------------------------------------------------------
// Reference speed to calculate speedup
#define REFSPEED 4000.0

// Default is that everything is vectorized (SSE | VECTORIZE_ALL)
// Otherwise you had to specify all VECTORIZE_* you want to use 
// (e.g. SIMD (SSE | VECTORIZE_BACKPROPAGATE | VECTORIZE_EVALUATE)) - To disable just add "& OFF"
#define SIMD (AVX | VECTORIZE_ALL)

#if SIMD & SSE
#define VEC_LENGTH 4
#define ALIGNMENT 16
typedef __m128 __mVec;
typedef __m128i __mVeci;

#define SIMD_INPUTSIZE (INPUTSIZE)
#define SIMD_NUMOUTPUT 20
#define SIMD_NUMHIDDEN (NUMHIDDEN+2)

static constexpr int simdNumHidden = getClosestDiv(NUMHIDDEN + 1, 2); 
static constexpr int simdNumOutput = getClosestDiv(NUMOUTPUT, 2);
static constexpr int simdNumInput = getClosestDiv(INPUTSIZE + 1, 2);

inline static __mVec _mVec_setr_ps(float a, float b, float c, float d, float e, float f, float g, float h) { return _mm_setr_ps(a, b, c, d); };
inline static __mVeci _mVec_setr_epi32(int a, int b, int c, int d, int e, int f, int g, int h) { return _mm_setr_epi32(a, b, c, d); };
inline static __mVeci _mVec_mullo_epi32(__mVeci a, __mVeci b) { return _mm_mullo_epi32(a, b); }; // Suppose SSE 4.1 is supported
inline static __mVeci _mVec_add_epi32(__mVeci a, __mVeci b) { return _mm_add_epi32(a, b); };
#elif SIMD & AVX

#define VEC_LENGTH 8
#define ALIGNMENT 32
typedef __m256 __mVec;
typedef __m256i __mVeci;

#define SIMD_INPUTSIZE (INPUTSIZE)
#define SIMD_NUMOUTPUT 20
#define SIMD_NUMHIDDEN (NUMHIDDEN+2)

inline static const int getClosestDivOf8(int i) { return ((i >> 3) + (bool)(i & 0b111)) << 3; }; // Make divisible by 8
static constexpr int simdNumHidden = getClosestDiv(NUMHIDDEN + 1, 3);
static constexpr int simdNumOutput = getClosestDiv(NUMOUTPUT, 3);
static constexpr int simdNumInput = getClosestDiv(INPUTSIZE + 1, 3);

#define _mm_setzero_ps _mm256_setzero_ps
#define _mm_mul_ps _mm256_mul_ps
#define _mm_div_ps _mm256_div_ps
#define _mm_add_ps _mm256_add_ps
#define _mm_sub_ps _mm256_sub_ps
#define _mm_cmp_ps _mm256_cmp_ps
#define _mm_andnot_ps _mm256_andnot_ps
#define _mm_and_ps _mm256_and_ps
#define _mm_load_ps _mm256_load_ps
#define _mm_loadu_ps _mm256_loadu_ps
#define _mm_store_ps _mm256_store_ps
#define _mVec_storeu_ps _mm256_storeu_ps
#define _mm_setr_ps _mm256_setr_ps
#define _mm_set_ps1 _mm256_set1_ps
// Integer operators
#define _mm_set_epi32 _mm256_set_epi32
#define _mm_set1_epi32 _mm256_set1_epi32
//#define _mm_add_epi32 _mm256_add_epi32
#define _mm_store_si128 _mm256_store_si256
#define _mm_storeu_si128 _mm256_storeu_si256
#define _mm_cvtps_epi32 _mm256_cvtps_epi32
#define _mm_cvttps_epi32 _mm256_cvttps_epi32

// While I don't have AVX 2 support ugly magic needs to be done
inline static __mVeci _mVec_mullo_epi32(__mVeci a, __mVeci b) { 
#ifdef __AVX2__
	return _mm256_mullo_epi32(a, b); 
#else
	__m128i vah = _mm256_extractf128_si256(a, 0);
	__m128i val = _mm256_extractf128_si256(a, 1);
	__m128i vbh = _mm256_extractf128_si256(b, 0);
	__m128i vbl = _mm256_extractf128_si256(b, 1);
	vah = _mm_mullo_epi32(vah, vbh);
	val = _mm_mullo_epi32(val, vbl);
	return _mm256_set_m128i(vah, val);
#endif
};
// While I don't have AVX 2 support ugly magic needs to be done
inline static __mVeci _mVec_add_epi32(__mVeci a, __mVeci b) {
#ifdef __AVX2__
	return _mm256_add_epi32(a, b);
#else
	__m128i vah = _mm256_extractf128_si256(a, 0);
	__m128i val = _mm256_extractf128_si256(a, 1);
	__m128i vbh = _mm256_extractf128_si256(b, 0);
	__m128i vbl = _mm256_extractf128_si256(b, 1);
	vah = _mm_add_epi32(vah, vbh);
	val = _mm_add_epi32(val, vbl);
	return _mm256_set_m128i(vah, val);
#endif
};

inline static __mVec _mVec_setr_ps(float a, float b, float c, float d, float e, float f, float g, float h) { return _mm256_setr_ps(a, b, c, d, e, f, g, h); };
inline static __mVeci _mVec_setr_epi32(int a, int b, int c, int d, int e, int f, int g, int h) { return _mm256_setr_epi32(a, b, c, d, e, f, g, h); };
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
	//union { float __declspec(align(ALIGNMENT)) inputNeurons[INPUTSIZE + VEC_LENGTH]; __mVec inputNeuronsVec[(INPUTSIZE + VEC_LENGTH) / VEC_LENGTH]; };
	float __declspec(align(ALIGNMENT)) inputNeurons[simdNumInput];
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
	union {	float __declspec(align(ALIGNMENT)) weightsInputHidden[(INPUTSIZE + 1) * (NUMHIDDEN + 1) + 1]; __mVec weightsInputHiddenVec[((INPUTSIZE + 1) * (NUMHIDDEN + 1) + 1) / VEC_LENGTH];};
	union {	float __declspec(align(ALIGNMENT)) weightsHiddenOutput[SIMD_NUMHIDDEN * NUMOUTPUT]; __mVec weightsHiddenOutputVec[SIMD_NUMHIDDEN * NUMOUTPUT / VEC_LENGTH];};
	// training data	
	union { float __declspec(align(ALIGNMENT)) deltaInputHidden[(INPUTSIZE + 1) * (NUMHIDDEN + 1) + 1]; __mVec deltaInputHiddenVec[((INPUTSIZE + 1) * (NUMHIDDEN + 1) + 1) / VEC_LENGTH]; };
	union { float __declspec(align(ALIGNMENT)) deltaHiddenOutput[SIMD_NUMHIDDEN * NUMOUTPUT+10]; __mVec deltaHiddenOutputVec[(SIMD_NUMHIDDEN * NUMOUTPUT + 10) / VEC_LENGTH]; };
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