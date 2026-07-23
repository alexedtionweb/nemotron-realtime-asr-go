// Package asr implements the streaming inference loop for the Nemotron 3.5
// ASR Streaming ONNX bundle: a cache-aware FastConformer encoder feeding a
// stateful RNN-T decoder+joint network.
package asr

import (
	"fmt"
	"log"
	"math"

	"github.com/alexedtionweb/nemotron-realtime-asr-go/internal/melspec"
	"github.com/alexedtionweb/nemotron-realtime-asr-go/internal/tokenizer"
	ort "github.com/yalue/onnxruntime_go"
)

func statsF32(data []float32) (minVal, maxVal, mean, rms float64) {
	if len(data) == 0 {
		return 0, 0, 0, 0
	}
	minVal, maxVal = math.Inf(1), math.Inf(-1)
	var sum, sumSq float64
	for _, v := range data {
		f := float64(v)
		if f < minVal {
			minVal = f
		}
		if f > maxVal {
			maxVal = f
		}
		sum += f
		sumSq += f * f
	}
	n := float64(len(data))
	return minVal, maxVal, sum / n, math.Sqrt(sumSq / n)
}

type Config struct {
	SampleRate            int
	NMels                 int
	NFFT                  int
	WindowSize            float64 // seconds
	WindowStride          float64 // seconds
	Preemphasis           float64
	SubsamplingFactor     int
	ChunkSizeOutputFrames int
	DropExtraPreEncoded   int
	NumEncoderLayers      int
	HiddenDim             int
	CacheLeftContext      int // cache_last_channel dim 2
	ConvContext           int // cache_last_time dim 3
	VocabSize             int // does NOT include blank
	BlankID               int
	DecoderStateDim       int // 640
	DecoderNumLayers      int // 2
	PromptIndex           int64
	Debug                 bool
}

// Token is a single emitted subword unit, tagged with the approximate
// wall-clock offset (in seconds, from the start of the session) of the
// encoder frame that produced it. Word boundaries are marked by the
// SentencePiece "▁" prefix convention, same as tokenizer.Detokenize.
type Token struct {
	ID      int32
	Piece   string
	TimeSec float64
}

type Model struct {
	cfg Config
	tok *tokenizer.Tokenizer
	mel *melspec.Extractor

	encoder *ort.DynamicAdvancedSession
	decoder *ort.DynamicAdvancedSession

	newChunkSamples int
	tailSamples     int
	hopSamples      int

	// Persistent encoder cache
	cacheLastChannel    []float32
	cacheLastTime       []float32
	cacheLastChannelLen int64
	audioTail           []float32

	// Persistent RNN-T prediction-network state
	decState1   []float32
	decState2   []float32
	lastToken   int32
	allTokenIDs []int32

	// Running count of encoded (post-subsampling) frames actually decoded
	// so far this session, used to derive absolute token timestamps.
	encFrameCounter int64

	// Reusable scratch buffers to guarantee zero heap allocations in hot loop
	combinedAudio []float32
	flatMel       []float32
	targetVal     []int32
	targetLenVal  []int32

	// Precomputed ONNX shapes
	shapeEncFrame        ort.Shape
	shapeTargets         ort.Shape
	shapeTargetLen       ort.Shape
	shapeDecState        ort.Shape
	shapeProcessedLen    ort.Shape
	shapeCacheChannel    ort.Shape
	shapeCacheTime       ort.Shape
	shapeCacheChannelLen ort.Shape
	shapePromptIdx       ort.Shape
}

func New(ortLibPath, encoderPath, decoderJointPath, tokenizerPath string, cfg Config) (*Model, error) {
	ort.SetSharedLibraryPath(ortLibPath)
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return nil, fmt.Errorf("initializing onnxruntime: %w", err)
		}
	}

	tok, err := tokenizer.Load(tokenizerPath)
	if err != nil {
		return nil, err
	}

	melCfg := melspec.DefaultConfig(cfg.SampleRate, cfg.NMels, cfg.NFFT, cfg.WindowSize, cfg.WindowStride, cfg.Preemphasis)
	mel := melspec.New(melCfg)

	encoder, err := ort.NewDynamicAdvancedSession(encoderPath,
		[]string{"processed_signal", "processed_signal_length", "cache_last_channel", "cache_last_time", "cache_last_channel_len", "prompt_index"},
		[]string{"encoded", "encoded_len", "cache_last_channel_next", "cache_last_time_next", "cache_last_channel_len_next"},
		nil)
	if err != nil {
		return nil, fmt.Errorf("loading encoder.onnx: %w", err)
	}

	decoder, err := ort.NewDynamicAdvancedSession(decoderJointPath,
		[]string{"encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"},
		[]string{"outputs", "prednet_lengths", "output_states_1", "output_states_2"},
		nil)
	if err != nil {
		encoder.Destroy()
		return nil, fmt.Errorf("loading decoder_joint.onnx: %w", err)
	}

	m := &Model{
		cfg:          cfg,
		tok:          tok,
		mel:          mel,
		encoder:      encoder,
		decoder:      decoder,
		targetVal:    make([]int32, 1),
		targetLenVal: []int32{1},
	}

	// Precompute static ONNX shapes
	m.shapeEncFrame = ort.NewShape(1, int64(cfg.HiddenDim), 1)
	m.shapeTargets = ort.NewShape(1, 1)
	m.shapeTargetLen = ort.NewShape(1)
	m.shapeDecState = ort.NewShape(int64(cfg.DecoderNumLayers), 1, int64(cfg.DecoderStateDim))
	m.shapeProcessedLen = ort.NewShape(1)
	m.shapeCacheChannel = ort.NewShape(int64(cfg.NumEncoderLayers), 1, int64(cfg.CacheLeftContext), int64(cfg.HiddenDim))
	m.shapeCacheTime = ort.NewShape(int64(cfg.NumEncoderLayers), 1, int64(cfg.HiddenDim), int64(cfg.ConvContext))
	m.shapeCacheChannelLen = ort.NewShape(1)
	m.shapePromptIdx = ort.NewShape(1)

	// Pre-allocate persistent state slice memory
	layers := cfg.NumEncoderLayers
	hidden := cfg.HiddenDim
	m.cacheLastChannel = make([]float32, layers*1*cfg.CacheLeftContext*hidden)
	m.cacheLastTime = make([]float32, layers*1*hidden*cfg.ConvContext)
	m.decState1 = make([]float32, cfg.DecoderNumLayers*1*cfg.DecoderStateDim)
	m.decState2 = make([]float32, cfg.DecoderNumLayers*1*cfg.DecoderStateDim)

	m.computeFraming()
	m.Reset()
	return m, nil
}

func (m *Model) computeFraming() {
	out := m.cfg.ChunkSizeOutputFrames + m.cfg.DropExtraPreEncoded
	melFramesNeeded := m.cfg.SubsamplingFactor*out - (m.cfg.SubsamplingFactor - 1)
	hop := int(math.Round(m.cfg.WindowStride * float64(m.cfg.SampleRate)))
	totalSamples := (melFramesNeeded - 1) * hop
	m.newChunkSamples = m.cfg.ChunkSizeOutputFrames * m.cfg.SubsamplingFactor * hop
	m.tailSamples = max(0, totalSamples-m.newChunkSamples)
	m.audioTail = make([]float32, m.tailSamples)
	m.hopSamples = hop
}

func (m *Model) ChunkSamples() int { return m.newChunkSamples }

// Reset clears state in-place using Go 1.21 clear() to avoid garbage collection.
func (m *Model) Reset() {
	clear(m.cacheLastChannel)
	clear(m.cacheLastTime)
	clear(m.audioTail)
	clear(m.decState1)
	clear(m.decState2)

	m.cacheLastChannelLen = 0
	m.lastToken = int32(m.cfg.BlankID)
	m.allTokenIDs = m.allTokenIDs[:0]
	m.encFrameCounter = 0
}

func (m *Model) SetPromptIndex(promptIndex int64) {
	m.cfg.PromptIndex = promptIndex
	m.Reset()
}

func (m *Model) Transcript() string {
	return m.tok.Detokenize(int32sToInts(m.allTokenIDs))
}

func (m *Model) FeedChunk(newSamples []float32) ([]Token, error) {
	if len(newSamples) != m.newChunkSamples {
		return nil, fmt.Errorf("FeedChunk expects exactly %d samples, got %d", m.newChunkSamples, len(newSamples))
	}

	// Reuse combined audio scratch buffer
	neededCap := len(m.audioTail) + len(newSamples)
	if cap(m.combinedAudio) < neededCap {
		m.combinedAudio = make([]float32, 0, neededCap)
	}
	combined := m.combinedAudio[:0]
	combined = append(combined, m.audioTail...)
	combined = append(combined, newSamples...)

	if m.cfg.Debug {
		mn, mx, mean, rms := statsF32(newSamples)
		log.Printf("[asr] raw audio chunk: n=%d min=%.4f max=%.4f mean=%.4f rms=%.4f", len(newSamples), mn, mx, mean, rms)
	}

	melFrames := m.mel.Compute(combined) // [NMels][T]
	T := 0
	if len(melFrames) > 0 {
		T = len(melFrames[0])
	}

	// Flatten mel features into reusable scratch buffer
	flatSize := m.cfg.NMels * T
	if cap(m.flatMel) < flatSize {
		m.flatMel = make([]float32, flatSize)
	} else {
		m.flatMel = m.flatMel[:flatSize]
	}

	for mIdx, row := range melFrames {
		copy(m.flatMel[mIdx*T:(mIdx+1)*T], row)
	}

	if m.cfg.Debug {
		mn, mx, mean, rms := statsF32(m.flatMel)
		log.Printf("[asr] mel features: shape=[1,%d,%d] min=%.4f max=%.4f mean=%.4f rms=%.4f", m.cfg.NMels, T, mn, mx, mean, rms)
	}

	// IMPORTANT: the first DropExtraPreEncoded output frames only exist to
	// give the encoder's conv/attention stack enough left-context (they were
	// computed from m.audioTail, i.e. audio that was already encoded and
	// decoded as part of the *previous* chunk). They must be dropped here —
	// decoding them re-feeds already-transcribed audio into the RNN-T and
	// produces duplicated/garbled words at every chunk boundary.
	encodedFrames, err := m.runEncoderStep(m.flatMel, T, m.cfg.DropExtraPreEncoded)
	if err != nil {
		return nil, err
	}
	baseFrame := m.encFrameCounter
	m.encFrameCounter += int64(len(encodedFrames))

	// Carry audio tail forward
	if m.tailSamples > 0 {
		start := max(0, len(newSamples)-m.tailSamples)
		copy(m.audioTail, newSamples[start:])
	}

	frameStepSec := float64(m.cfg.SubsamplingFactor*m.hopSamples) / float64(m.cfg.SampleRate)

	var emitted []Token
	for i, encFrame := range encodedFrames {
		frameTime := float64(baseFrame+int64(i)) * frameStepSec
		toks, err := m.greedyDecodeFrame(encFrame, frameTime)
		if err != nil {
			return nil, err
		}
		emitted = append(emitted, toks...)
	}
	for _, t := range emitted {
		m.allTokenIDs = append(m.allTokenIDs, t.ID)
	}
	return emitted, nil
}

func (m *Model) runEncoderStep(melFlat []float32, T int, drop int) ([][]float32, error) {
	hidden := m.cfg.HiddenDim

	processedSignal, err := ort.NewTensor(ort.NewShape(1, int64(m.cfg.NMels), int64(T)), melFlat)
	if err != nil {
		return nil, err
	}
	defer processedSignal.Destroy()

	processedLen, err := ort.NewTensor(m.shapeProcessedLen, []int64{int64(T)})
	if err != nil {
		return nil, err
	}
	defer processedLen.Destroy()

	cacheChannel, err := ort.NewTensor(m.shapeCacheChannel, m.cacheLastChannel)
	if err != nil {
		return nil, err
	}
	defer cacheChannel.Destroy()

	cacheTime, err := ort.NewTensor(m.shapeCacheTime, m.cacheLastTime)
	if err != nil {
		return nil, err
	}
	defer cacheTime.Destroy()

	cacheChannelLen, err := ort.NewTensor(m.shapeCacheChannelLen, []int64{m.cacheLastChannelLen})
	if err != nil {
		return nil, err
	}
	defer cacheChannelLen.Destroy()

	promptIdx, err := ort.NewTensor(m.shapePromptIdx, []int64{m.cfg.PromptIndex})
	if err != nil {
		return nil, err
	}
	defer promptIdx.Destroy()

	inputs := []ort.Value{processedSignal, processedLen, cacheChannel, cacheTime, cacheChannelLen, promptIdx}
	outputs := make([]ort.Value, 5)
	if err := m.encoder.Run(inputs, outputs); err != nil {
		return nil, fmt.Errorf("running encoder: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	encodedTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected type for encoder 'encoded' output")
	}
	encShape := encodedTensor.GetShape()
	encT := int(encShape[2])
	encData := encodedTensor.GetData()

	if m.cfg.Debug {
		mn, mx, mean, rms := statsF32(encData)
		log.Printf("[asr] encoder output: shape=%v min=%.4f max=%.4f mean=%.4f rms=%.4f", encShape, mn, mx, mean, rms)
	}

	cacheChannelNext, ok := outputs[2].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected type for 'cache_last_channel_next'")
	}
	copy(m.cacheLastChannel, cacheChannelNext.GetData())

	cacheTimeNext, ok := outputs[3].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected type for 'cache_last_time_next'")
	}
	copy(m.cacheLastTime, cacheTimeNext.GetData())

	cacheChannelLenNext, ok := outputs[4].(*ort.Tensor[int64])
	if !ok {
		return nil, fmt.Errorf("unexpected type for 'cache_last_channel_len_next'")
	}
	if data := cacheChannelLenNext.GetData(); len(data) > 0 {
		m.cacheLastChannelLen = data[0]
	}

	drop = min(drop, encT)
	kept := encT - drop

	if m.cfg.Debug {
		log.Printf("[asr] dropping %d pre-encoded context frame(s), keeping %d/%d", drop, kept, encT)
	}

	// Allocate frames in a single contiguous block for cache locality
	backing := make([]float32, kept*hidden)
	frames := make([][]float32, kept)
	for t := drop; t < encT; t++ {
		outIdx := t - drop
		vec := backing[outIdx*hidden : (outIdx+1)*hidden]
		for h := range hidden {
			vec[h] = encData[h*encT+t]
		}
		frames[outIdx] = vec
	}

	return frames, nil
}

// greedyDecodeFrame runs the RNN-T label loop for a single encoder frame.
// Zero-allocation path for state & tensor inputs. frameTime is the
// approximate session-relative time (seconds) this encoder frame covers,
// stamped onto every token emitted from it.
func (m *Model) greedyDecodeFrame(encFrame []float32, frameTime float64) ([]Token, error) {
	const maxSymbolsPerStep = 10
	var emitted []Token

	for step := range maxSymbolsPerStep {
		// Create tensors directly over persistent memory slices (no append copies)
		encTensor, err := ort.NewTensor(m.shapeEncFrame, encFrame)
		if err != nil {
			return nil, err
		}

		m.targetVal[0] = m.lastToken
		targets, err := ort.NewTensor(m.shapeTargets, m.targetVal)
		if err != nil {
			encTensor.Destroy()
			return nil, err
		}

		targetLen, err := ort.NewTensor(m.shapeTargetLen, m.targetLenVal)
		if err != nil {
			encTensor.Destroy()
			targets.Destroy()
			return nil, err
		}

		state1, err := ort.NewTensor(m.shapeDecState, m.decState1)
		if err != nil {
			encTensor.Destroy()
			targets.Destroy()
			targetLen.Destroy()
			return nil, err
		}

		state2, err := ort.NewTensor(m.shapeDecState, m.decState2)
		if err != nil {
			encTensor.Destroy()
			targets.Destroy()
			targetLen.Destroy()
			state1.Destroy()
			return nil, err
		}

		inputs := []ort.Value{encTensor, targets, targetLen, state1, state2}
		outputs := make([]ort.Value, 4)
		runErr := m.decoder.Run(inputs, outputs)

		encTensor.Destroy()
		targets.Destroy()
		targetLen.Destroy()
		state1.Destroy()
		state2.Destroy()

		if runErr != nil {
			for _, o := range outputs {
				if o != nil {
					o.Destroy()
				}
			}
			return nil, fmt.Errorf("running decoder_joint: %w", runErr)
		}

		logitsTensor, ok := outputs[0].(*ort.Tensor[float32])
		if !ok {
			for _, o := range outputs {
				if o != nil {
					o.Destroy()
				}
			}
			return nil, fmt.Errorf("unexpected type for decoder_joint 'outputs'")
		}

		logits := logitsTensor.GetData()
		best, bestVal := 0, float32(math.Inf(-1))
		secondBest, secondVal := 0, float32(math.Inf(-1))
		for i, v := range logits {
			if v > bestVal {
				secondBest, secondVal = best, bestVal
				best, bestVal = i, v
			} else if v > secondVal {
				secondBest, secondVal = i, v
			}
		}

		if m.cfg.Debug && step == 0 {
			blankVal := float32(math.NaN())
			if m.cfg.BlankID < len(logits) {
				blankVal = logits[m.cfg.BlankID]
			}
			log.Printf("[asr] decode: nLogits=%d argmax=%d(val=%.4f) runnerUp=%d(val=%.4f) blankID=%d blankVal=%.4f",
				len(logits), best, bestVal, secondBest, secondVal, m.cfg.BlankID, blankVal)
		}

		outState1, ok1 := outputs[2].(*ort.Tensor[float32])
		outState2, ok2 := outputs[3].(*ort.Tensor[float32])

		// Direct copy into persistent decState without allocation
		if ok1 && ok2 {
			copy(m.decState1, outState1.GetData())
			copy(m.decState2, outState2.GetData())
		}

		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}

		if best == m.cfg.BlankID {
			break
		}

		emitted = append(emitted, Token{
			ID:      int32(best),
			Piece:   m.tok.IDToPiece(best),
			TimeSec: frameTime,
		})
		m.lastToken = int32(best)
	}

	return emitted, nil
}

func (m *Model) Close() {
	if m.encoder != nil {
		m.encoder.Destroy()
	}
	if m.decoder != nil {
		m.decoder.Destroy()
	}

}

func (m *Model) TranscribeOffline(samples []float32) (string, error) {
	m.Reset()
	if len(samples) == 0 {
		return "", nil
	}

	for offset := 0; offset < len(samples); offset += m.newChunkSamples {
		end := min(offset+m.newChunkSamples, len(samples))
		chunk := samples[offset:end]
		if len(chunk) < m.newChunkSamples {
			padded := make([]float32, m.newChunkSamples)
			copy(padded, chunk)
			chunk = padded
		}
		if _, err := m.FeedChunk(chunk); err != nil {
			return "", fmt.Errorf("processing audio at sample %d: %w", offset, err)
		}
	}
	return m.Transcript(), nil
}

func int32sToInts(xs []int32) []int {
	out := make([]int, len(xs))
	for i, x := range xs {
		out[i] = int(x)
	}
	return out
}
