package melspec

import "testing"

func TestNumFramesMatchesModelTestVector(t *testing.T) {
	cfg := DefaultConfig(16000, 128, 512, 0.025, 0.01, 0.97)
	e := New(cfg)
	got := e.NumFrames(10240) // 1280 tail + 8960 new samples
	if got != 65 {
		t.Fatalf("NumFrames(10240) = %d, want 65", got)
	}
}

func TestComputeShape(t *testing.T) {
	cfg := DefaultConfig(16000, 128, 512, 0.025, 0.01, 0.97)
	e := New(cfg)
	samples := make([]float32, 10240)
	for i := range samples {
		samples[i] = float32(i%100) / 100.0
	}
	feats := e.Compute(samples)
	if len(feats) != 128 {
		t.Fatalf("got %d mel bins, want 128", len(feats))
	}
	if len(feats[0]) != 65 {
		t.Fatalf("got %d frames, want 65", len(feats[0]))
	}
}
