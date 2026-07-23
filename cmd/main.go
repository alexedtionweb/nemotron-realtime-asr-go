package main

import (
	_ "embed"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/alexedtionweb/nemotron-realtime-asr-go/internal/asr"
	"github.com/gorilla/websocket"
	ort "github.com/yalue/onnxruntime_go"
)

//go:embed index.html
var indexHTML []byte

var upgrader = websocket.Upgrader{
	// Allow any origin for local development
	CheckOrigin: func(r *http.Request) bool { return true },
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

type serverConfig struct {
	modelDir string
	ortLib   string
	model    modelConfig
}

type clientMessage struct {
	Type     string `json:"type"`
	Language string `json:"language"`
}

type transcriptMessage struct {
	Type  string `json:"type"`
	Text  string `json:"text,omitempty"`
	Error string `json:"error,omitempty"`
}

type modelPool struct {
	cfg  serverConfig
	mu   sync.Mutex
	idle []*asr.Model
}

func newModelPool(cfg serverConfig) *modelPool {
	return &modelPool{cfg: cfg}
}

func (p *modelPool) acquire(language string) (*asr.Model, error) {
	prompt, ok := p.cfg.model.PromptDictionary[language]
	if !ok {
		return nil, fmt.Errorf("unsupported language %q", language)
	}
	p.mu.Lock()
	if len(p.idle) > 0 {
		model := p.idle[len(p.idle)-1]
		p.idle = p.idle[:len(p.idle)-1]
		p.mu.Unlock()
		model.SetPromptIndex(prompt)
		return model, nil
	}
	p.mu.Unlock()
	return p.cfg.newModel(language)
}

func (p *modelPool) release(model *asr.Model) {
	if model == nil {
		return
	}
	model.Reset()
	p.mu.Lock()
	if len(p.idle) == 0 {
		p.idle = append(p.idle, model)
		p.mu.Unlock()
		return
	}
	p.mu.Unlock()
	model.Close()
}

func (p *modelPool) close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, model := range p.idle {
		model.Close()
	}
	p.idle = nil
}

func main() {
	// 1. Better Entrypoint: Use flags instead of hardcoded absolute paths
	modelDir := flag.String("model-dir", "/home/alex/.cache/huggingface/hub/models--tonythethompson--Nemotron-3.5-ASR-Streaming-0.6B-ONNX/snapshots/3a2c8c48cf093c890740577d9467f835ca95fdbd/", "Path to the downloaded ONNX model directory")
	ortLib := flag.String("ort-lib", "/usr/local/lib/libonnxruntime.so", "Path to the ONNX Runtime shared library")
	listenAddr := flag.String("addr", ":8081", "HTTP server address to listen on")
	flag.Parse()

	cfg, err := loadServerConfig(*modelDir, *ortLib)
	if err != nil {
		log.Fatalf("Failed to load server config: %v", err)
	}

	ort.SetSharedLibraryPath(cfg.ortLib)
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("Initialize ONNX Runtime: %v", err)
	}
	defer ort.DestroyEnvironment()

	pool := newModelPool(cfg)
	if model, err := pool.acquire("auto"); err != nil {
		log.Fatalf("Preload auto language model: %v", err)
	} else {
		pool.release(model)
	}
	defer pool.close()

	// 2. Standard Library HTTP Multiplexer (No Fiber)
	mux := http.NewServeMux()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexHTML)
	})

	mux.HandleFunc("/languages", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(availableLanguages(cfg.model.PromptDictionary)); err != nil {
			log.Printf("Failed to encode languages: %v", err)
		}
	})

	mux.HandleFunc("/ws/transcribe", func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("WebSocket upgrade failed: %v", err)
			return
		}
		handleTranscription(conn, pool)
	})

	log.Printf("Server listening on http://localhost%s", *listenAddr)
	if err := http.ListenAndServe(*listenAddr, mux); err != nil {
		log.Fatal(err)
	}
}

func availableLanguages(prompts map[string]int64) []string {
	byPrompt := make(map[int64]string)
	for code, prompt := range prompts {
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

func loadServerConfig(modelDir, ortLib string) (serverConfig, error) {
	if _, err := os.Stat(ortLib); err != nil {
		return serverConfig{}, fmt.Errorf("ONNX Runtime library %q: %w", ortLib, err)
	}
	b, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return serverConfig{}, fmt.Errorf("read model config: %w", err)
	}
	var mc modelConfig
	if err := json.Unmarshal(b, &mc); err != nil {
		return serverConfig{}, fmt.Errorf("parse model config: %w", err)
	}
	return serverConfig{modelDir: modelDir, ortLib: ortLib, model: mc}, nil
}

func (c serverConfig) newModel(language string) (*asr.Model, error) {
	prompt, ok := c.model.PromptDictionary[language]
	if !ok {
		return nil, fmt.Errorf("unsupported language %q", language)
	}
	cacheLeft := 0
	if len(c.model.CacheShapes.CacheLastChannel) == 4 {
		cacheLeft = c.model.CacheShapes.CacheLastChannel[2]
	}
	return asr.New(c.ortLib,
		filepath.Join(c.modelDir, "encoder.onnx"),
		filepath.Join(c.modelDir, "decoder_joint.onnx"),
		filepath.Join(c.modelDir, "tokenizer.model"),
		asr.Config{
			SampleRate: c.model.SampleRate, NMels: c.model.NMels, NFFT: c.model.Preprocessor.NFFT,
			WindowSize: c.model.Preprocessor.WindowSize, WindowStride: c.model.Preprocessor.WindowStride,
			Preemphasis: c.model.Preprocessor.Preemph, SubsamplingFactor: c.model.SubsamplingFactor,
			ChunkSizeOutputFrames: c.model.ChunkSizeOutputFrames, DropExtraPreEncoded: c.model.DropExtraPreEncoded,
			NumEncoderLayers: c.model.NumEncoderLayers, HiddenDim: c.model.HiddenDim, CacheLeftContext: cacheLeft,
			ConvContext: c.model.ConvContext, VocabSize: c.model.VocabSize, BlankID: c.model.BlankID,
			DecoderStateDim: 640, DecoderNumLayers: 2, PromptIndex: prompt,
		})
}

// 3. Updated for Gorilla WebSocket API (which conveniently maps 1:1 with Fiber's syntax)
func handleTranscription(conn *websocket.Conn, pool *modelPool) {
	defer conn.Close()
	log.Printf("Microphone client connected: %s", conn.RemoteAddr())

	var model *asr.Model
	var pending []float32
	defer func() { pool.release(model) }()
	for {
		messageType, payload, err := conn.ReadMessage()
		if err != nil {
			return
		}
		switch messageType {
		case websocket.TextMessage:
			var message clientMessage
			if err := json.Unmarshal(payload, &message); err != nil {
				writeMessage(conn, transcriptMessage{Type: "error", Error: "invalid control message"})
				continue
			}
			switch message.Type {
			case "start":
				if model != nil {
					pool.release(model)
					model = nil
				}
				language := message.Language
				if language == "" {
					language = "auto"
				}
				model, err = pool.acquire(language)
				pending = pending[:0]
				if err != nil {
					writeMessage(conn, transcriptMessage{Type: "error", Error: err.Error()})
					model = nil
					continue
				}
				writeMessage(conn, transcriptMessage{Type: "ready"})
			case "end":
				if model != nil {
					if len(pending) > 0 {
						chunk := make([]float32, model.ChunkSamples())
						copy(chunk, pending)
						if _, err := model.FeedChunk(chunk); err != nil {
							writeMessage(conn, transcriptMessage{Type: "error", Error: err.Error()})
						}
					}
					writeMessage(conn, transcriptMessage{Type: "final", Text: model.Transcript()})
					pool.release(model)
					model = nil
					pending = pending[:0]
				}
			}
		case websocket.BinaryMessage:
			if model == nil {
				writeMessage(conn, transcriptMessage{Type: "error", Error: "send a start message before audio"})
				continue
			}
			if len(payload)%2 != 0 {
				writeMessage(conn, transcriptMessage{Type: "error", Error: "audio must be PCM16 little-endian"})
				continue
			}
			pending = append(pending, pcm16ToFloat32(payload)...)
			for len(pending) >= model.ChunkSamples() {
				if _, err := model.FeedChunk(pending[:model.ChunkSamples()]); err != nil {
					writeMessage(conn, transcriptMessage{Type: "error", Error: err.Error()})
					break
				}
				pending = pending[model.ChunkSamples():]
				writeMessage(conn, transcriptMessage{Type: "partial", Text: model.Transcript()})
			}
		}
	}
}

func pcm16ToFloat32(data []byte) []float32 {
	samples := make([]float32, len(data)/2)
	for i := range samples {
		samples[i] = float32(int16(binary.LittleEndian.Uint16(data[i*2:]))) / math.MaxInt16
	}
	return samples
}

func writeMessage(conn *websocket.Conn, message transcriptMessage) {
	b, _ := json.Marshal(message)
	if err := conn.WriteMessage(websocket.TextMessage, b); err != nil {
		log.Printf("write WebSocket message: %v", err)
	}
}
