// Package melspec computes log-mel spectrogram features matching NVIDIA
// NeMo's AudioToMelSpectrogramPreprocessor, driven by the model's own
// config.json values (sample_rate, n_mels, window_size, window_stride,
// n_fft, preemph). NeMo builds its mel filters via librosa.filters.mel(...)
// with librosa's defaults, so this matches those specifically: the Slaney
// mel scale (htk=False) with Slaney-style area-normalized filters
// (norm='slaney'), a periodic Hann window, and a 2**-24 log-guard epsilon.
package melspec

import "math"

type Config struct {
	SampleRate    int
	NMels         int
	NFFT          int
	WinLength     int // samples
	HopLength     int // samples
	Preemphasis   float32
	LogGuardValue float32
}

// DefaultConfig builds a Config from the model's config.json fields.
func DefaultConfig(sampleRate, nMels, nFFT int, windowSize, windowStride, preemph float64) Config {
	return Config{
		SampleRate:    sampleRate,
		NMels:         nMels,
		NFFT:          nFFT,
		WinLength:     int(math.Round(windowSize * float64(sampleRate))),
		HopLength:     int(math.Round(windowStride * float64(sampleRate))),
		Preemphasis:   float32(preemph),
		LogGuardValue: float32(math.Pow(2, -24)),
	}
}

type Extractor struct {
	cfg      Config
	window   []float32   // Hann window, length WinLength
	melBank  [][]float32 // [NMels][NFFT/2+1]
	fftOrder int
}

// New builds an Extractor, precomputing the analysis window and mel filterbank.
func New(cfg Config) *Extractor {
	e := &Extractor{cfg: cfg}
	e.window = hannWindow(cfg.WinLength)
	e.melBank = melFilterbank(cfg.NMels, cfg.NFFT, cfg.SampleRate)
	e.fftOrder = cfg.NFFT
	return e
}

// NumFrames returns how many STFT frames `numSamples` of audio produce,
// matching torch.stft's center=True framing (num_samples/hop + 1).
func (e *Extractor) NumFrames(numSamples int) int {
	if numSamples <= 0 {
		return 0
	}
	return numSamples/e.cfg.HopLength + 1
}

// Compute returns log-mel features as [NMels][numFrames] (channel-major,
// matching the ONNX `processed_signal` layout of [batch, n_mels, time]).
func (e *Extractor) Compute(samples []float32) [][]float32 {
	pre := preemphasize(samples, e.cfg.Preemphasis)
	padded := centerPad(pre, e.cfg.NFFT)

	numFrames := e.NumFrames(len(samples))
	feats := make([][]float32, e.cfg.NMels)
	for m := range feats {
		feats[m] = make([]float32, numFrames)
	}

	frame := make([]complex128, e.fftOrder)
	powerSpec := make([]float64, e.cfg.NFFT/2+1)

	for f := range numFrames {
		start := f * e.cfg.HopLength
		for i := 0; i < e.fftOrder; i++ {
			var s float32
			idx := start + i
			if i < e.cfg.WinLength && idx < len(padded) {
				s = padded[idx] * e.window[i]
			}
			frame[i] = complex(float64(s), 0)
		}
		fft(frame)
		for k := range powerSpec {
			re := real(frame[k])
			im := imag(frame[k])
			powerSpec[k] = re*re + im*im
		}
		for m := 0; m < e.cfg.NMels; m++ {
			var sum float64
			row := e.melBank[m]
			for k, w := range row {
				if w != 0 {
					sum += float64(w) * powerSpec[k]
				}
			}
			feats[m][f] = float32(math.Log(sum + float64(e.cfg.LogGuardValue)))
		}
	}
	return feats
}

func preemphasize(x []float32, coef float32) []float32 {
	if len(x) == 0 {
		return x
	}
	out := make([]float32, len(x))
	out[0] = x[0]
	for i := 1; i < len(x); i++ {
		out[i] = x[i] - coef*x[i-1]
	}
	return out
}

// centerPad reflects nFFT/2 samples onto each end, matching torch.stft's
// default center=True, pad_mode="reflect".
func centerPad(x []float32, nFFT int) []float32 {
	padAmt := nFFT / 2
	out := make([]float32, 0, len(x)+2*padAmt)
	for i := padAmt; i >= 1; i-- {
		idx := i
		if idx >= len(x) {
			idx = len(x) - 1
		}
		if idx < 0 {
			idx = 0
		}
		out = append(out, x[idx])
	}
	out = append(out, x...)
	for i := range padAmt {
		idx := max(len(x)-2-i, 0)
		out = append(out, x[idx])
	}
	return out
}

// hannWindow returns a periodic Hann window (matching torch.stft's default
// window convention: 0.5 - 0.5*cos(2*pi*n/N), using N rather than N-1 in the
// denominator).
func hannWindow(n int) []float32 {
	w := make([]float32, n)
	for i := 0; i < n; i++ {
		w[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n)))
	}
	return w
}

// melFilterbank builds a triangular mel filterbank matching librosa's
// defaults (which is what NeMo's FilterbankFeatures calls under the hood):
// the Slaney mel scale (htk=False) with Slaney-style area normalization
// (norm='slaney'), and ramps computed by linear interpolation directly in
// Hz-space against FFT bin center frequencies (not snapped to integer bins).
func melFilterbank(nMels, nFFT, sampleRate int) [][]float32 {
	nBins := nFFT/2 + 1
	fMin, fMax := 0.0, float64(sampleRate)/2.0

	// FFT bin center frequencies: linspace(0, sr/2, nBins).
	fftFreqs := make([]float64, nBins)
	for k := range nBins {
		fftFreqs[k] = float64(k) * fMax / float64(nBins-1)
	}

	// nMels+2 mel-spaced points, converted to Hz via the Slaney mel scale.
	melMin := hzToMelSlaney(fMin)
	melMax := hzToMelSlaney(fMax)
	melPoints := make([]float64, nMels+2)
	hzPoints := make([]float64, nMels+2)
	for i := range melPoints {
		melPoints[i] = melMin + (melMax-melMin)*float64(i)/float64(nMels+1)
		hzPoints[i] = melToHzSlaney(melPoints[i])
	}

	fdiff := make([]float64, nMels+1)
	for i := range fdiff {
		fdiff[i] = hzPoints[i+1] - hzPoints[i]
	}

	bank := make([][]float32, nMels)
	for m := range nMels {
		row := make([]float32, nBins)
		for k := range nBins {
			lower := (fftFreqs[k] - hzPoints[m]) / fdiff[m]
			upper := (hzPoints[m+2] - fftFreqs[k]) / fdiff[m+1]
			w := math.Min(lower, upper)
			if w < 0 {
				w = 0
			}
			row[k] = float32(w)
		}
		// Slaney-style area normalization: scale each filter so its
		// integral is constant regardless of band width.
		enorm := 2.0 / (hzPoints[m+2] - hzPoints[m])
		for k := range row {
			row[k] = float32(float64(row[k]) * enorm)
		}
		bank[m] = row
	}
	return bank
}

// hzToMelSlaney / melToHzSlaney implement librosa's default (htk=False)
// mel scale: linear below 1kHz, logarithmic above it.
func hzToMelSlaney(hz float64) float64 {
	const (
		fMin     = 0.0
		fSp      = 200.0 / 3.0
		minLogHz = 1000.0
	)
	minLogMel := (minLogHz - fMin) / fSp
	logstep := math.Log(6.4) / 27.0
	if hz < minLogHz {
		return (hz - fMin) / fSp
	}
	return minLogMel + math.Log(hz/minLogHz)/logstep
}

func melToHzSlaney(mel float64) float64 {
	const (
		fMin     = 0.0
		fSp      = 200.0 / 3.0
		minLogHz = 1000.0
	)
	minLogMel := (minLogHz - fMin) / fSp
	logstep := math.Log(6.4) / 27.0
	if mel < minLogMel {
		return fMin + fSp*mel
	}
	return minLogHz * math.Exp(logstep*(mel-minLogMel))
}

// fft performs an in-place iterative radix-2 Cooley-Tukey FFT. len(x) must
// be a power of two (true for n_fft=512 here).
func fft(x []complex128) {
	n := len(x)
	if n <= 1 {
		return
	}
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			x[i], x[j] = x[j], x[i]
		}
	}
	for length := 2; length <= n; length <<= 1 {
		ang := -2 * math.Pi / float64(length)
		wlen := complex(math.Cos(ang), math.Sin(ang))
		for i := 0; i < n; i += length {
			w := complex(1.0, 0.0)
			for j := 0; j < length/2; j++ {
				u := x[i+j]
				v := x[i+j+length/2] * w
				x[i+j] = u + v
				x[i+j+length/2] = u - v
				w *= wlen
			}
		}
	}
}
