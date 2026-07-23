// Command nemotron-asr transcribes a 16kHz WAV file using the Nemotron 3.5
// ASR Streaming ONNX bundle, feeding audio through the model in the same
// fixed-size chunks a real-time caller would use (so this doubles as a
// correctness check for the streaming loop even though it runs on a file).
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"nemotron-asr/internal/asr"
	"nemotron-asr/internal/wav"

	ort "github.com/yalue/onnxruntime_go"
)

// modelConfig mirrors the fields we need out of the repo's config.json.
type modelConfig struct {
	SampleRate            int              `json:"sample_rate"`
	NMels                 int              `json:"n_mels"`
	SubsamplingFactor     int              `json:"subsampling_factor"`
	AttContextSize        [2]int           `json:"att_context_size"`
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
		CacheLastChannel []int `json:"cache_last_channel"` // [layers,1,left,hidden]
	} `json:"cache_shapes"`
}

// resolveONNXRuntimeLibPath finds the ONNX Runtime shared library to load,
// preferring (in order): the explicit -ort-lib flag, the ONNXRUNTIME_LIB_PATH
// env var, then a handful of common install locations.
func resolveONNXRuntimeLibPath(flagValue string) (string, error) {
	if flagValue != "" {
		return flagValue, nil
	}
	if envPath := os.Getenv("ONNXRUNTIME_LIB_PATH"); envPath != "" {
		return envPath, nil
	}
	candidates := []string{
		"/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib",
		"/usr/local/opt/onnxruntime/lib/libonnxruntime.dylib",
		"/opt/homebrew/lib/libonnxruntime.dylib",
		"/usr/local/lib/libonnxruntime.dylib",
		"/usr/local/lib/libonnxruntime.so",
		"/usr/lib/libonnxruntime.so",
	}
	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
	}
	return "", fmt.Errorf("no ONNX Runtime library found; pass -ort-lib, set ONNXRUNTIME_LIB_PATH, or install one (macOS: brew install onnxruntime)")
}

// InitializeONNXRuntime points ONNX Runtime at libPath and starts its
// environment. Safe to call even if asr.New() will also touch the runtime:
// asr.New() checks ort.IsInitialized() before re-initializing.
func InitializeONNXRuntime(libPath string) error {
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
	}
	return nil
}

func main() {
	modelDir := flag.String("model-dir", "/home/alex/.cache/huggingface/hub/models--tonythethompson--Nemotron-3.5-ASR-Streaming-0.6B-ONNX/snapshots/3a2c8c48cf093c890740577d9467f835ca95fdbd/", "directory containing config.json, encoder.onnx, decoder_joint.onnx, tokenizer.model")
	audioPath := flag.String("audio", "/home/alex/advanced-realtime-alexander-pipeline/speech.wav", "path to a 16kHz mono (or auto-resampled) WAV file to transcribe")
	ortLib := flag.String("ort-lib", "", "path to the ONNX Runtime shared library (libonnxruntime.so/.dylib/.dll); falls back to ONNXRUNTIME_LIB_PATH env var, then common install paths")
	language := flag.String("language", "es", "language/locale code from config.json's prompt_dictionary, or \"auto\"")
	debug := flag.Bool("debug", false, "log audio/mel/encoder/logit stats to stderr for troubleshooting")
	offline := flag.Bool("offline", false, "transcribe the complete local file in one call (the streaming encoder is windowed internally)")
	flag.Parse()

	if *modelDir == "" || *audioPath == "" {
		fmt.Fprintln(os.Stderr, "usage: nemotron-asr -model-dir <dir> -audio <file.wav> [-ort-lib <path/to/libonnxruntime.so>] [-language en]")
		flag.PrintDefaults()
		os.Exit(2)
	}

	libPath, err := resolveONNXRuntimeLibPath(*ortLib)
	if err != nil {
		log.Fatalf("%v", err)
	}
	if err := InitializeONNXRuntime(libPath); err != nil {
		log.Fatalf("Error initializing ONNX Runtime: %v\nHint: install ONNX Runtime (macOS: brew install onnxruntime) or pass -ort-lib", err)
	}

	cfgBytes, err := os.ReadFile(filepath.Join(*modelDir, "config.json"))
	if err != nil {
		log.Fatalf("reading config.json: %v", err)
	}
	var mc modelConfig
	if err := json.Unmarshal(cfgBytes, &mc); err != nil {
		log.Fatalf("parsing config.json: %v", err)
	}

	promptIndex, ok := mc.PromptDictionary[*language]
	if !ok {
		log.Fatalf("language %q not found in config.json prompt_dictionary", *language)
	}

	cacheLeft := 0
	if len(mc.CacheShapes.CacheLastChannel) == 4 {
		cacheLeft = mc.CacheShapes.CacheLastChannel[2]
	}

	cfg := asr.Config{
		SampleRate:            mc.SampleRate,
		NMels:                 mc.NMels,
		NFFT:                  mc.Preprocessor.NFFT,
		WindowSize:            mc.Preprocessor.WindowSize,
		WindowStride:          mc.Preprocessor.WindowStride,
		Preemphasis:           mc.Preprocessor.Preemph,
		SubsamplingFactor:     mc.SubsamplingFactor,
		ChunkSizeOutputFrames: mc.ChunkSizeOutputFrames,
		DropExtraPreEncoded:   mc.DropExtraPreEncoded,
		NumEncoderLayers:      mc.NumEncoderLayers,
		HiddenDim:             mc.HiddenDim,
		CacheLeftContext:      cacheLeft,
		ConvContext:           mc.ConvContext,
		VocabSize:             mc.VocabSize,
		BlankID:               mc.BlankID,
		DecoderStateDim:       640,
		DecoderNumLayers:      2,
		PromptIndex:           promptIndex,
		Debug:                 *debug,
	}

	log.Printf("model dir:      %s", *modelDir)
	log.Printf("audio file:     %s", *audioPath)
	log.Printf("ort lib:        %s", libPath)
	log.Printf("language:       %s (prompt_index=%d)", *language, promptIndex)
	log.Printf("sample_rate=%d n_mels=%d blank_id=%d vocab_size=%d", mc.SampleRate, mc.NMels, mc.BlankID, mc.VocabSize)

	model, err := asr.New(
		libPath,
		filepath.Join(*modelDir, "encoder.onnx"),
		filepath.Join(*modelDir, "decoder_joint.onnx"),
		filepath.Join(*modelDir, "tokenizer.model"),
		cfg,
	)
	if err != nil {
		log.Fatalf("loading model: %v", err)
	}
	defer model.Close()

	samples, err := wav.ReadFile(*audioPath, mc.SampleRate)
	if err != nil {
		log.Fatalf("reading audio: %v", err)
	}
	log.Printf("loaded audio: %d samples (%.2fs at %dHz)", len(samples), float64(len(samples))/float64(mc.SampleRate), mc.SampleRate)
	if len(samples) == 0 {
		log.Fatalf("audio file decoded to zero samples — check the WAV file isn't empty/corrupt")
	}

	if *offline {
		transcript, err := model.TranscribeOffline(samples)
		if err != nil {
			log.Fatalf("offline transcription: %v", err)
		}
		log.Printf("[offline mode: complete file processed through encoder windows]")
		fmt.Println(transcript)
		return
	}

	chunkSize := model.ChunkSamples()
	log.Printf("chunk size: %d samples (%.0fms)", chunkSize, 1000*float64(chunkSize)/float64(mc.SampleRate))

	chunkNum := 0
	totalTokens := 0
	for offset := 0; offset < len(samples); offset += chunkSize {
		end := offset + chunkSize
		var chunk []float32
		if end <= len(samples) {
			chunk = samples[offset:end]
		} else {
			// Zero-pad the final, shorter chunk.
			chunk = make([]float32, chunkSize)
			copy(chunk, samples[offset:])
		}
		ids, err := model.FeedChunk(chunk)
		if err != nil {
			log.Fatalf("feeding chunk at offset %d: %v", offset, err)
		}
		chunkNum++
		totalTokens += len(ids)
		log.Printf("chunk %d: emitted %d token(s), running transcript so far: %q", chunkNum, len(ids), model.Transcript())
	}

	transcript := model.Transcript()
	if transcript == "" {
		log.Printf("WARNING: %d chunks processed, 0 tokens ever emitted (every prediction was blank_id=%d) — this points at a real pipeline issue (silent/near-silent audio, wrong -language, or a mismatch between the mel features and what the model expects), not just quiet speech", chunkNum, mc.BlankID)
	}
	fmt.Println(transcript)
}
