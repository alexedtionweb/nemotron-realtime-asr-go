package asr

import "testing"

// TestFramingMatchesModelTestVector checks our derived chunk-framing
// constants against the exact numbers the model repo's config.json ships
// as its own test_input/test_output vector:
//
//	mel_shape: [1, 128, 65], mel_length: 65  ->  encoded_shape: [1, 1024, 7]
//
// i.e. 65 mel frames in, 7 encoder frames out. The two extra pre-encoded
// frames are input context and are already excluded by the exported graph.
func TestFramingMatchesModelTestVector(t *testing.T) {
	cfg := Config{
		SampleRate:            16000,
		WindowStride:          0.01,
		SubsamplingFactor:     8,
		ChunkSizeOutputFrames: 7,
		DropExtraPreEncoded:   2,
	}
	m := &Model{cfg: cfg}
	m.computeFraming()

	wantMelFrames := 65
	out := cfg.ChunkSizeOutputFrames + cfg.DropExtraPreEncoded
	gotMelFrames := cfg.SubsamplingFactor*out - (cfg.SubsamplingFactor - 1)
	if gotMelFrames != wantMelFrames {
		t.Fatalf("mel frames needed = %d, want %d", gotMelFrames, wantMelFrames)
	}

	// New audio per chunk should be exactly 560ms at 16kHz (README's
	// documented streaming chunk size for this model).
	wantNewChunkSamples := 8960
	if m.newChunkSamples != wantNewChunkSamples {
		t.Fatalf("newChunkSamples = %d, want %d", m.newChunkSamples, wantNewChunkSamples)
	}

	wantTailSamples := 1280 // 80ms carried-over overlap
	if m.tailSamples != wantTailSamples {
		t.Fatalf("tailSamples = %d, want %d", m.tailSamples, wantTailSamples)
	}
}
