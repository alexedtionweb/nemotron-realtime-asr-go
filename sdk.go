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
	debug    bool
}

// SetDebug enables or disables debug logging for future sessions.
func (e *Engine) SetDebug(enabled bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.debug = enabled
	for _, m := range e.idle {
		m.SetDebug(enabled)
	}
}

// Word is a single finalized word with session-relative timestamps, in the
// same spirit as the "words" array returned by Deepgram/Google STT/OpenAI's
// realtime transcription APIs.
type Word struct {
	Text  string  `json:"word"`
	Start float64 `json:"start_sec"`
	End   float64 `json:"end_sec"`
}

// Result is returned from every WritePCM16/Finalize call. It mirrors the
// shape used by industry streaming STT APIs: a running transcript, the
// words that became final during this call, and an is_final flag for
// utterance boundaries.
//
// Because Nemotron's cache-aware streaming encoder never revises tokens it
// has already emitted (unlike re-decoding approaches), every word in
// NewWords is genuinely final the moment it's returned — it will not change
// on subsequent calls. IsFinal instead marks the *session* boundary (set
// only by Finalize), since this SDK has no VAD/endpointing model to detect
// mid-stream utterance breaks on its own.
type Result struct {
	Type       string `json:"type"` // "partial" | "final"
	Transcript string `json:"transcript"`
	NewWords   []Word `json:"new_words,omitempty"`
	IsFinal    bool   `json:"is_final"`

	// Diarization is always nil. Nemotron 3.5 Streaming is a
	// transcription-only RNN-T model — it has no speaker-embedding or
	// clustering head, so it cannot produce speaker labels. Real
	// diarization needs a separate model (e.g. NeMo's Sortformer / speaker
	// diarization pipeline) run alongside this one, with its speaker turns
	// aligned against Word.Start/Word.End. This field is kept in the schema
	// so clients written against Deepgram/Google-style payloads don't have
	// to special-case this API, but it's never populated here.
	Diarization any `json:"diarization"`
}

// Session represents a single active audio stream being transcribed.
type Session struct {
	engine  *Engine
	model   *asr.Model
	pending []float32

	// OnResult, if set, is invoked with the partial Result produced by every
	// Write call (the io.Writer entry point). This mirrors the callback/event
	// style used by mainstream streaming STT SDKs (Deepgram, AssemblyAI,
	// Google) and lets callers pipe audio in with a standard io.Copy without
	// juggling return values.
	OnResult func(*Result)

	oddByte    byte
	hasOddByte bool

	words         []Word
	openWord      strings.Builder
	openStart     float64
	openActive    bool
	lastTokenTime float64
}

// Write implements io.Writer over the raw little-endian PCM16 audio stream,
// making a Session a drop-in sink for io.Copy, bufio, and the rest of the
// standard streaming plumbing. Each call runs inference on any newly
// completed chunks and, if OnResult is set, delivers the partial Result.
// It always reports len(p) bytes consumed on success, since partial PCM16
// frames are buffered internally until the next write.
func (s *Session) Write(p []byte) (int, error) {
	// Reassemble a 2-byte PCM16 sample that may have been split across
	// writes (common with io.Copy's fixed-size buffers).
	buf := p
	if s.hasOddByte {
		buf = append([]byte{s.oddByte}, p...)
		s.hasOddByte = false
	}
	if len(buf)%2 != 0 {
		s.oddByte = buf[len(buf)-1]
		s.hasOddByte = true
		buf = buf[:len(buf)-1]
	}

	res, err := s.WritePCM16(buf)
	if err != nil {
		return 0, err
	}
	if s.OnResult != nil {
		s.OnResult(res)
	}
	return len(p), nil
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
	// Fallback used when prompt_dictionary is absent (e.g. Nemotron 3.5 HF models).
	DefaultPromptID int64 `json:"default_prompt_id"`
	NumPrompts      int   `json:"num_prompts"`
	Preprocessor    struct {
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
// When the model config has no prompt_dictionary (e.g. Nemotron 3.5 HF format),
// returns ["auto"] backed by default_prompt_id.
func (e *Engine) AvailableLanguages() []string {
	if len(e.cfg.PromptDictionary) == 0 {
		return []string{"auto"}
	}
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
	var prompt int64
	if len(e.cfg.PromptDictionary) == 0 {
		// Model uses default_prompt_id instead of a language dictionary
		// (Nemotron 3.5 HF config format). Accept "auto" or "" as aliases for
		// the default, and allow raw numeric prompt IDs as strings.
		switch language {
		case "auto", "":
			prompt = e.cfg.DefaultPromptID
		default:
			return nil, fmt.Errorf("unsupported language %q (model has no prompt dictionary; use \"auto\")", language)
		}
	} else {
		var ok bool
		prompt, ok = e.cfg.PromptDictionary[language]
		if !ok {
			return nil, fmt.Errorf("unsupported language %q", language)
		}
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
			DecoderStateDim: 640, DecoderNumLayers: 2, PromptIndex: prompt, Debug: e.debug,
		})
	if err != nil {
		return nil, err
	}

	return &Session{engine: e, model: m}, nil
}

// ingestTokens folds newly-decoded subword tokens into whole words, using
// the SentencePiece "▁" prefix to detect word starts. It returns the words
// that became final (closed) during this call and updates session state for
// the word still being built, if any.
func (s *Session) ingestTokens(tokens []asr.Token) []Word {
	var closed []Word
	for _, t := range tokens {
		isWordStart := strings.HasPrefix(t.Piece, "\u2581")
		piece := strings.TrimPrefix(t.Piece, "\u2581")

		if isWordStart {
			if w, ok := s.flushOpenWord(); ok {
				closed = append(closed, w)
			}
			s.openStart = t.TimeSec
			s.openActive = true
		} else if !s.openActive {
			s.openStart = t.TimeSec
			s.openActive = true
		}

		s.openWord.WriteString(piece)
		s.lastTokenTime = t.TimeSec
	}
	return closed
}

func (s *Session) flushOpenWord() (Word, bool) {
	if !s.openActive || s.openWord.Len() == 0 {
		s.openWord.Reset()
		s.openActive = false
		return Word{}, false
	}
	w := Word{Text: s.openWord.String(), Start: s.openStart, End: s.lastTokenTime}
	s.words = append(s.words, w)
	s.openWord.Reset()
	s.openActive = false
	return w, true
}

// WritePCM16 accepts raw little-endian PCM16 bytes, buffers them, runs
// inference on every full chunk that has accumulated, and returns a
// structured partial result: the running transcript plus any words that
// became final during this call.
func (s *Session) WritePCM16(data []byte) (*Result, error) {
	if len(data)%2 != 0 {
		return nil, fmt.Errorf("audio must be PCM16 little-endian (even length)")
	}

	// Convert bytes to float32 and append to session buffer
	for i := 0; i < len(data); i += 2 {
		val := float32(int16(binary.LittleEndian.Uint16(data[i:]))) / math.MaxInt16
		s.pending = append(s.pending, val)
	}

	chunkSize := s.model.ChunkSamples()
	var newWords []Word

	// Consume full chunks via an index instead of reslicing s.pending
	// forward. Reslicing (s.pending = s.pending[chunkSize:]) slides the
	// slice header through an ever-growing backing array over the life of a
	// long stream, so the array is never reclaimed — a per-session memory
	// leak that adds up fast under many concurrent users.
	consumed := 0
	for len(s.pending)-consumed >= chunkSize {
		tokens, err := s.model.FeedChunk(s.pending[consumed : consumed+chunkSize])
		if err != nil {
			return nil, err
		}
		newWords = append(newWords, s.ingestTokens(tokens)...)
		consumed += chunkSize
	}
	if consumed > 0 {
		// Move the residual (< chunkSize) samples back to the front of the
		// same backing array and shrink the length, keeping capacity bounded.
		n := copy(s.pending, s.pending[consumed:])
		s.pending = s.pending[:n]
	}

	return &Result{
		Type:       "partial",
		Transcript: s.model.Transcript(),
		NewWords:   newWords,
		IsFinal:    false,
	}, nil
}

// Finalize processes any remaining trailing audio, closes out whatever word
// was still being built, and returns the final result for the session.
func (s *Session) Finalize() (*Result, error) {
	var newWords []Word

	if len(s.pending) > 0 {
		chunk := make([]float32, s.model.ChunkSamples())
		copy(chunk, s.pending)
		tokens, err := s.model.FeedChunk(chunk)
		if err != nil {
			return nil, err
		}
		newWords = append(newWords, s.ingestTokens(tokens)...)
		s.pending = s.pending[:0]
	}

	if w, ok := s.flushOpenWord(); ok {
		newWords = append(newWords, w)
	}

	return &Result{
		Type:       "final",
		Transcript: s.model.Transcript(),
		NewWords:   newWords,
		IsFinal:    true,
	}, nil
}

// Close resets the model state and returns it to the engine pool.
// MUST be called when finished to prevent memory leaks.
func (s *Session) Close() {
	if s.model == nil {
		return
	}
	s.model.Reset()
	s.pending = s.pending[:0]
	s.hasOddByte = false

	s.engine.mu.Lock()
	defer s.engine.mu.Unlock()
	s.engine.idle = append(s.engine.idle, s.model)
	s.model = nil
}
