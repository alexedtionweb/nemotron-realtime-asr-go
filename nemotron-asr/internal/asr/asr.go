// Package asr implements the streaming inference loop for the Nemotron 3.5
// ASR Streaming ONNX bundle: a cache-aware FastConformer encoder feeding a
// stateful RNN-T decoder+joint network.
//
// Pipeline per audio chunk:
//
//	raw audio -> log-mel features -> encoder.onnx (carries attention/conv
//	cache across chunks) -> drop the "extra pre-encoded" warm-up frames ->
//	greedy RNN-T decode over decoder_joint.onnx (carries LSTM prediction
//	state across the whole stream, token by token).
package asr

import (
	"fmt"
	"log"
	"math"

	ort "github.com/yalue/onnxruntime_go"

	"nemotron-asr/internal/melspec"
	"nemotron-asr/internal/tokenizer"
)

// statsF32 returns (min, max, mean, rms) for quick sanity-checking that a
// buffer isn't silent/flat/NaN.
func statsF32(data []float32) (min, max, mean, rms float64) {
	if len(data) == 0 {
		return 0, 0, 0, 0
	}
	min, max = math.Inf(1), math.Inf(-1)
	var sum, sumSq float64
	for _, v := range data {
		f := float64(v)
		if f < min {
			min = f
		}
		if f > max {
			max = f
		}
		sum += f
		sumSq += f * f
	}
	n := float64(len(data))
	mean = sum / n
	rms = math.Sqrt(sumSq / n)
	return
}

// Config mirrors the fields we need from the model's config.json.
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
	DecoderNumLayers      int // 2 (the leading "2" in input_states_1/2 shape)
	PromptIndex           int64
	Debug                 bool // log signal stats (audio/mel/encoder/logits) to stderr
}

// Model wraps the loaded ONNX sessions plus all persistent streaming state.
type Model struct {
	cfg Config
	tok *tokenizer.Tokenizer
	mel *melspec.Extractor

	encoder *ort.DynamicAdvancedSession
	decoder *ort.DynamicAdvancedSession

	// Derived chunk-framing constants (see computeFraming).
	newChunkSamples int
	tailSamples     int

	// Persistent encoder cache (cache-aware streaming state).
	cacheLastChannel    []float32 // [layers,1,cacheLeft,hidden]
	cacheLastTime       []float32 // [layers,1,hidden,convContext]
	cacheLastChannelLen int64
	audioTail           []float32 // last `tailSamples` raw samples, carried between chunks

	// Persistent RNN-T prediction-network state.
	decState1 []float32 // [decoderNumLayers,1,decoderStateDim]
	decState2 []float32
	lastToken int32

	allTokenIDs []int32
}

// New loads the ONNX Runtime shared library plus both model graphs and the
// tokenizer, and prepares a fresh streaming Model.
//
// ortLibPath must point at the onnxruntime shared library for your platform
// (e.g. libonnxruntime.so / .dylib / .dll) — see README.md for how to get one.
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
		cfg:     cfg,
		tok:     tok,
		mel:     mel,
		encoder: encoder,
		decoder: decoder,
	}
	m.computeFraming()
	m.Reset()
	return m, nil
}

// computeFraming derives the raw-sample chunk size and carried-over tail
// length from the subsampling geometry, so that each streaming step's mel
// input always yields exactly cfg.ChunkSizeOutputFrames genuinely-new
// encoder frames after dropping the warm-up frames.
//
// The FastConformer subsampling front-end is 3 stride-2/kernel-3/pad-1 conv
// layers (2^3 = SubsamplingFactor). For that geometry the minimal mel input
// length producing `out` output frames is out*2-1 applied three times, i.e.
// mel_in = SubsamplingFactor*out - (SubsamplingFactor-1).
func (m *Model) computeFraming() {
	out := m.cfg.ChunkSizeOutputFrames + m.cfg.DropExtraPreEncoded
	melFramesNeeded := m.cfg.SubsamplingFactor*out - (m.cfg.SubsamplingFactor - 1)
	hop := int(math.Round(m.cfg.WindowStride * float64(m.cfg.SampleRate)))
	totalSamples := (melFramesNeeded - 1) * hop
	m.newChunkSamples = m.cfg.ChunkSizeOutputFrames * m.cfg.SubsamplingFactor * hop
	m.tailSamples = totalSamples - m.newChunkSamples
	if m.tailSamples < 0 {
		m.tailSamples = 0
	}
}

// ChunkSamples returns how many new raw audio samples FeedChunk expects.
func (m *Model) ChunkSamples() int { return m.newChunkSamples }

// Reset clears all streaming state (encoder cache, decoder LSTM state,
// audio tail, and accumulated transcript) so the Model can be reused for a
// new utterance/stream.
func (m *Model) Reset() {
	layers := m.cfg.NumEncoderLayers
	hidden := m.cfg.HiddenDim
	m.cacheLastChannel = make([]float32, layers*1*m.cfg.CacheLeftContext*hidden)
	m.cacheLastTime = make([]float32, layers*1*hidden*m.cfg.ConvContext)
	m.cacheLastChannelLen = 0
	m.audioTail = make([]float32, m.tailSamples)

	m.decState1 = make([]float32, m.cfg.DecoderNumLayers*1*m.cfg.DecoderStateDim)
	m.decState2 = make([]float32, m.cfg.DecoderNumLayers*1*m.cfg.DecoderStateDim)
	m.lastToken = int32(m.cfg.BlankID)
	m.allTokenIDs = nil
}

// SetPromptIndex selects the language prompt used by the next encoder call
// and clears all state from the previous utterance. Language selection is an
// encoder input, so one loaded model can safely be reused across languages.
func (m *Model) SetPromptIndex(promptIndex int64) {
	m.cfg.PromptIndex = promptIndex
	m.Reset()
}

// Transcript detokenizes all tokens emitted so far.
func (m *Model) Transcript() string {
	return m.tok.Detokenize(int32sToInts(m.allTokenIDs))
}

// FeedChunk pushes ChunkSamples() worth of new 16kHz mono audio through the
// encoder and then greedily decodes any newly available encoder frames. Pad
// the final, shorter-than-usual chunk of a stream with trailing zeros before
// calling. Returns the token ids emitted by this call (already appended to
// the running transcript).
func (m *Model) FeedChunk(newSamples []float32) ([]int32, error) {
	if len(newSamples) != m.newChunkSamples {
		return nil, fmt.Errorf("FeedChunk expects exactly %d samples, got %d", m.newChunkSamples, len(newSamples))
	}

	combined := make([]float32, 0, len(m.audioTail)+len(newSamples))
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
	flat := make([]float32, m.cfg.NMels*T)
	for mIdx, row := range melFrames {
		copy(flat[mIdx*T:(mIdx+1)*T], row)
	}

	if m.cfg.Debug {
		mn, mx, mean, rms := statsF32(flat)
		log.Printf("[asr] mel features: shape=[1,%d,%d] min=%.4f max=%.4f mean=%.4f rms=%.4f", m.cfg.NMels, T, mn, mx, mean, rms)
	}

	// drop_extra_pre_encoded is input context, not encoder output to discard.
	// The 65-frame input includes two extra frames so the exported graph emits
	// the requested seven newly-encoded frames. The graph already returns
	// those seven frames (as confirmed by config.json's test vector); dropping
	// two more here loses speech at every chunk boundary.
	encodedIDs, err := m.runEncoderStep(flat, T, 0)
	if err != nil {
		return nil, err
	}

	// Carry the tail of this chunk's raw audio forward for the next call.
	if m.tailSamples > 0 {
		start := len(newSamples) - m.tailSamples
		if start < 0 {
			start = 0
		}
		m.audioTail = append(m.audioTail[:0], newSamples[start:]...)
	}

	var emitted []int32
	for _, encFrame := range encodedIDs {
		ids, err := m.greedyDecodeFrame(encFrame)
		if err != nil {
			return nil, err
		}
		emitted = append(emitted, ids...)
	}
	m.allTokenIDs = append(m.allTokenIDs, emitted...)
	return emitted, nil
}

// runEncoderStep runs encoder.onnx on one chunk's mel features, updates the
// persistent cache state, optionally drops leading frames, and returns the
// kept encoder frames as a slice of [hidden]float32 vectors in time order.
// The current Nemotron export already excludes its extra pre-encoded input
// context from `encoded`, so production callers pass drop=0.
func (m *Model) runEncoderStep(melFlat []float32, T int, drop int) ([][]float32, error) {
	hidden := m.cfg.HiddenDim
	layers := m.cfg.NumEncoderLayers

	processedSignal, err := ort.NewTensor(ort.NewShape(1, int64(m.cfg.NMels), int64(T)), melFlat)
	if err != nil {
		return nil, err
	}
	defer processedSignal.Destroy()
	processedLen, err := ort.NewTensor(ort.NewShape(1), []int64{int64(T)})
	if err != nil {
		return nil, err
	}
	defer processedLen.Destroy()
	cacheChannel, err := ort.NewTensor(ort.NewShape(int64(layers), 1, int64(m.cfg.CacheLeftContext), int64(hidden)), m.cacheLastChannel)
	if err != nil {
		return nil, err
	}
	defer cacheChannel.Destroy()
	cacheTime, err := ort.NewTensor(ort.NewShape(int64(layers), 1, int64(hidden), int64(m.cfg.ConvContext)), m.cacheLastTime)
	if err != nil {
		return nil, err
	}
	defer cacheTime.Destroy()
	cacheChannelLen, err := ort.NewTensor(ort.NewShape(1), []int64{m.cacheLastChannelLen})
	if err != nil {
		return nil, err
	}
	defer cacheChannelLen.Destroy()
	promptIdx, err := ort.NewTensor(ort.NewShape(1), []int64{m.cfg.PromptIndex})
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
	encShape := encodedTensor.GetShape() // [1, hidden, encT]
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
	m.cacheLastChannel = append(m.cacheLastChannel[:0], cacheChannelNext.GetData()...)

	cacheTimeNext, ok := outputs[3].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected type for 'cache_last_time_next'")
	}
	m.cacheLastTime = append(m.cacheLastTime[:0], cacheTimeNext.GetData()...)

	cacheChannelLenNext, ok := outputs[4].(*ort.Tensor[int64])
	if !ok {
		return nil, fmt.Errorf("unexpected type for 'cache_last_channel_len_next'")
	}
	if data := cacheChannelLenNext.GetData(); len(data) > 0 {
		m.cacheLastChannelLen = data[0]
	}

	// Drop `drop` warm-up frames from the front; keep the genuinely-new tail frames.
	if drop > encT {
		drop = encT
	}
	kept := encT - drop
	frames := make([][]float32, 0, kept)
	for t := drop; t < encT; t++ {
		vec := make([]float32, hidden)
		for h := 0; h < hidden; h++ {
			// encData layout is [1, hidden, encT] row-major: index = h*encT + t
			vec[h] = encData[h*encT+t]
		}
		frames = append(frames, vec)
	}
	return frames, nil
}

// greedyDecodeFrame runs the RNN-T label loop for a single encoder time
// frame, emitting zero or more tokens (blank stops the loop for this frame).
func (m *Model) greedyDecodeFrame(encFrame []float32) ([]int32, error) {
	const maxSymbolsPerStep = 10
	hidden := len(encFrame)
	var emitted []int32

	for step := 0; step < maxSymbolsPerStep; step++ {
		encTensor, err := ort.NewTensor(ort.NewShape(1, int64(hidden), 1), append([]float32{}, encFrame...))
		if err != nil {
			return nil, err
		}
		targets, err := ort.NewTensor(ort.NewShape(1, 1), []int32{m.lastToken})
		if err != nil {
			encTensor.Destroy()
			return nil, err
		}
		targetLen, err := ort.NewTensor(ort.NewShape(1), []int32{1})
		if err != nil {
			encTensor.Destroy()
			targets.Destroy()
			return nil, err
		}
		state1, err := ort.NewTensor(ort.NewShape(int64(m.cfg.DecoderNumLayers), 1, int64(m.cfg.DecoderStateDim)), append([]float32{}, m.decState1...))
		if err != nil {
			encTensor.Destroy()
			targets.Destroy()
			targetLen.Destroy()
			return nil, err
		}
		state2, err := ort.NewTensor(ort.NewShape(int64(m.cfg.DecoderNumLayers), 1, int64(m.cfg.DecoderStateDim)), append([]float32{}, m.decState2...))
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
		var newState1, newState2 []float32
		if ok1 && ok2 {
			// Copy out of the tensors *before* destroying them below.
			newState1 = append(newState1, outState1.GetData()...)
			newState2 = append(newState2, outState2.GetData()...)
		}

		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}

		if best == m.cfg.BlankID {
			break // move on to the next encoder frame; state/lastToken unchanged
		}
		emitted = append(emitted, int32(best))
		m.lastToken = int32(best)
		if ok1 && ok2 {
			m.decState1 = newState1
			m.decState2 = newState2
		}
	}
	return emitted, nil
}

// Close releases the ONNX Runtime sessions.
func (m *Model) Close() {
	if m.encoder != nil {
		m.encoder.Destroy()
	}
	if m.decoder != nil {
		m.decoder.Destroy()
	}
}

// TranscribeOffline transcribes a complete, already-available utterance.
//
// Despite its name, the exported encoder is a streaming graph: its output is
// capped at ChunkSizeOutputFrames (seven frames for the current bundle), even
// if a much longer mel tensor is supplied. Therefore an offline file must
// still be passed through the encoder's fixed-size windows and caches. This
// method owns that loop so callers that have a whole file do not accidentally
// lose all but the first encoder window.
func (m *Model) TranscribeOffline(samples []float32) (string, error) {
	m.Reset()
	if len(samples) == 0 {
		return "", nil
	}

	for offset := 0; offset < len(samples); offset += m.newChunkSamples {
		end := offset + m.newChunkSamples
		chunk := samples[offset:min(end, len(samples))]
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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func int32sToInts(xs []int32) []int {
	out := make([]int, len(xs))
	for i, x := range xs {
		out[i] = int(x)
	}
	return out
}
