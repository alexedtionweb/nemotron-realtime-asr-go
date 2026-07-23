// Package nemotron provides a high-level SDK for running real-time streaming
// ASR using the Nemotron 3.5 ONNX model.
package nemotron

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/alexedtionweb/nemotron-realtime-asr-go/internal/asr"
	ort "github.com/yalue/onnxruntime_go"
)

// Engine represents the global ASR inference engine and model pool.
type Engine struct {
	modelDir string
	ortLib   string
	cfg      modelConfig
	mu       sync.Mutex
	idle     []*asr.Model
}

// Session represents a single active audio stream being transcribed.
type Session struct {
	engine  *Engine
	model   *asr.Model
	pending []float32
}

type modelConfig struct {
	SampleRate            int              `json:"sample_rate"`
	NMels                 int              `json:"n_mels"`
	SubsamplingFactor     int              `json:"subsampling_factor"`
	ChunkSizeOutputFrames int              `json:"chunk_size_output_frames"`
	DropExtraPreEncoded   int              `json:"drop_extra_pre_encoded"`
	NumEncoderLayers      int              `json:"num_encoder_layers"`
	HiddenDim             int              `json:"hidden_dim"`
	ConvContext           int              `json:"conv_context"`
	VocabSize             int              `json:"vocab_size"`
	BlankID               int              `json:"blank_id"`
	PromptDictionary      map[string]int64 `json:"prompt_dictionary"`
	Preprocessor          struct {
		WindowSize   float64 `json:"window_size"`
		WindowStride float64 `json:"window_stride"`
		NFFT         int     `json:"n_fft"`
		Preemph      float64 `json:"preemph"`
	} `json:"preprocessor"`
	CacheShapes struct {
		CacheLastChannel []int `json:"cache_last_channel"`
	} `json:"cache_shapes"`
}

// NewEngine initializes the ONNX Runtime and prepares the model pool.
func NewEngine(modelDir, ortLib string) (*Engine, error) {
	if _, err := os.Stat(ortLib); err != nil {
		return nil, fmt.Errorf("ONNX Runtime library not found: %w", err)
	}

	b, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("read model config: %w", err)
	}

	var mc modelConfig
	if err := json.Unmarshal(b, &mc); err != nil {
		return nil, fmt.Errorf("parse model config: %w", err)
	}

	ort.SetSharedLibraryPath(ortLib)
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return nil, fmt.Errorf("initialize ONNX Runtime: %w", err)
		}
	}

	engine := &Engine{
		modelDir: modelDir,
		ortLib:   ortLib,
		cfg:      mc,
	}

	// Pre-warm the auto language model to eliminate startup latency on first use
	if session, err := engine.NewSession("auto"); err == nil {
		session.Close()
	}

	return engine, nil
}

// Close gracefully destroys the model pool and the ONNX environment.
func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, m := range e.idle {
		m.Close()
	}
	e.idle = nil
	ort.DestroyEnvironment()
}

// AvailableLanguages returns a sorted list of supported language codes.
func (e *Engine) AvailableLanguages() []string {
	byPrompt := make(map[int64]string)
	for code, prompt := range e.cfg.PromptDictionary {
		current, exists := byPrompt[prompt]
		if !exists || languageCodeScore(code) < languageCodeScore(current) ||
			(languageCodeScore(code) == languageCodeScore(current) && code < current) {
			byPrompt[prompt] = code
		}
	}
	result := make([]string, 0, len(byPrompt))
	for _, code := range byPrompt {
		result = append(result, code)
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i] == "auto" {
			return true
		}
		if result[j] == "auto" {
			return false
		}
		return result[i] < result[j]
	})
	return result
}

func languageCodeScore(code string) int {
	if code == "auto" {
		return -1
	}
	if !strings.Contains(code, "-") {
		return 0
	}
	return 1
}

// NewSession requests a fresh transcription session from the pool.
func (e *Engine) NewSession(language string) (*Session, error) {
	prompt, ok := e.cfg.PromptDictionary[language]
	if !ok {
		return nil, fmt.Errorf("unsupported language %q", language)
	}

	e.mu.Lock()
	if len(e.idle) > 0 {
		m := e.idle[len(e.idle)-1]
		e.idle = e.idle[:len(e.idle)-1]
		e.mu.Unlock()
		m.SetPromptIndex(prompt)
		return &Session{engine: e, model: m}, nil
	}
	e.mu.Unlock()

	// Build a new model if the pool is empty
	cacheLeft := 0
	if len(e.cfg.CacheShapes.CacheLastChannel) == 4 {
		cacheLeft = e.cfg.CacheShapes.CacheLastChannel[2]
	}

	m, err := asr.New(e.ortLib,
		filepath.Join(e.modelDir, "encoder.onnx"),
		filepath.Join(e.modelDir, "decoder_joint.onnx"),
		filepath.Join(e.modelDir, "tokenizer.model"),
		asr.Config{
			SampleRate: e.cfg.SampleRate, NMels: e.cfg.NMels, NFFT: e.cfg.Preprocessor.NFFT,
			WindowSize: e.cfg.Preprocessor.WindowSize, WindowStride: e.cfg.Preprocessor.WindowStride,
			Preemphasis: e.cfg.Preprocessor.Preemph, SubsamplingFactor: e.cfg.SubsamplingFactor,
			ChunkSizeOutputFrames: e.cfg.ChunkSizeOutputFrames, DropExtraPreEncoded: e.cfg.DropExtraPreEncoded,
			NumEncoderLayers: e.cfg.NumEncoderLayers, HiddenDim: e.cfg.HiddenDim, CacheLeftContext: cacheLeft,
			ConvContext: e.cfg.ConvContext, VocabSize: e.cfg.VocabSize, BlankID: e.cfg.BlankID,
			DecoderStateDim: 640, DecoderNumLayers: 2, PromptIndex: prompt,
		})
	if err != nil {
		return nil, err
	}

	return &Session{engine: e, model: m}, nil
}

// WritePCM16 accepts raw little-endian PCM16 bytes, buffers them, runs inference
// if enough frames are collected, and returns the current partial transcript.
func (s *Session) WritePCM16(data []byte) (string, error) {
	if len(data)%2 != 0 {
		return "", fmt.Errorf("audio must be PCM16 little-endian (even length)")
	}

	// Convert bytes to float32 and append to session buffer
	for i := 0; i < len(data); i += 2 {
		val := float32(int16(binary.LittleEndian.Uint16(data[i:]))) / math.MaxInt16
		s.pending = append(s.pending, val)
	}

	chunkSize := s.model.ChunkSamples()
	var err error

	for len(s.pending) >= chunkSize {
		_, err = s.model.FeedChunk(s.pending[:chunkSize])
		if err != nil {
			return "", err
		}
		s.pending = s.pending[chunkSize:]
	}

	return s.model.Transcript(), nil
}

// Finalize processes any remaining trailing audio and returns the final text.
func (s *Session) Finalize() (string, error) {
	if len(s.pending) > 0 {
		chunk := make([]float32, s.model.ChunkSamples())
		copy(chunk, s.pending)
		if _, err := s.model.FeedChunk(chunk); err != nil {
			return "", err
		}
		s.pending = s.pending[:0] // Clear buffer
	}
	return s.model.Transcript(), nil
}

// Close resets the model state and returns it to the engine pool.
// MUST be called when finished to prevent memory leaks.
func (s *Session) Close() {
	if s.model == nil {
		return
	}
	s.model.Reset()
	s.pending = s.pending[:0]

	s.engine.mu.Lock()
	defer s.engine.mu.Unlock()
	s.engine.idle = append(s.engine.idle, s.model)
	s.model = nil
}
