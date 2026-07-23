package melspec

import (
	"math"
)

type Config struct {
	SampleRate    int
	NMels         int
	NFFT          int
	WinLength     int // samples
	HopLength     int // samples
	Preemphasis   float32
	LogGuardValue float32
}

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

// melFilter represents a sparse mel filter row to avoid N^2 zero-multiplications
type melFilter struct {
	startBin int
	weights  []float32
}

type Extractor struct {
	cfg        Config
	window     []float32
	melFilters []melFilter
	fftPlan    *fftPlan
}

// New builds an Extractor, precomputing the analysis window, sparse filters, and FFT plan.
// Extractor is completely thread-safe for concurrent Compute() calls.
func New(cfg Config) *Extractor {
	e := &Extractor{
		cfg:     cfg,
		window:  hannWindow(cfg.WinLength),
		fftPlan: newFFTPlan(cfg.NFFT),
	}

	// Build the dense filterbank, then compress it into a sparse representation
	denseBank := melFilterbank(cfg.NMels, cfg.NFFT, cfg.SampleRate)
	e.melFilters = make([]melFilter, cfg.NMels)

	for m, row := range denseBank {
		start := -1
		end := -1
		for k, w := range row {
			if w > 0 {
				if start == -1 {
					start = k
				}
				end = k
			}
		}
		if start != -1 {
			weights := make([]float32, end-start+1)
			copy(weights, row[start:end+1])
			e.melFilters[m] = melFilter{
				startBin: start,
				weights:  weights,
			}
		}
	}

	return e
}

func (e *Extractor) NumFrames(numSamples int) int {
	if numSamples <= 0 {
		return 0
	}
	return numSamples/e.cfg.HopLength + 1
}

// Compute returns log-mel features. Optimized to guarantee zero heap allocations
// in the hot path beyond the required output matrix.
func (e *Extractor) Compute(samples []float32) [][]float32 {
	numSamples := len(samples)
	numFrames := e.NumFrames(numSamples)
	if numFrames == 0 {
		return nil
	}

	// 1. Single Contiguous Allocation for the 2D output
	// Keeps cache locality perfect (matching ONNX expectations).
	backing := make([]float32, e.cfg.NMels*numFrames)
	feats := make([][]float32, e.cfg.NMels)
	for m := range e.cfg.NMels {
		feats[m] = backing[m*numFrames : (m+1)*numFrames]
	}

	// 2. Local Scratch Buffers (No heap escape, highly efficient)
	frame := make([]complex128, e.cfg.NFFT)
	powerSpec := make([]float64, e.cfg.NFFT/2+1)

	padAmt := e.cfg.NFFT / 2
	maxPaddedIdx := numSamples + e.cfg.NFFT
	logGuard := float64(e.cfg.LogGuardValue)

	// 3. Main processing loop
	for f := range numFrames {
		start := f * e.cfg.HopLength

		// Fill complex frame using virtual padding & pre-emphasis (zero-copy)
		for i := range e.cfg.NFFT {
			var s float32
			paddedIdx := start + i
			if i < e.cfg.WinLength && paddedIdx < maxPaddedIdx {
				s = readPaddedPreemph(samples, paddedIdx, padAmt, e.cfg.Preemphasis)
				s *= e.window[i]
			}
			frame[i] = complex(float64(s), 0)
		}

		// In-place FFT
		e.fftPlan.exec(frame)

		// Compute power spectrum
		for k := range powerSpec {
			re := real(frame[k])
			im := imag(frame[k])
			powerSpec[k] = re*re + im*im
		}

		// Apply Sparse Mel Filterbank
		for m, filter := range e.melFilters {
			var sum float64
			for i, w := range filter.weights {
				sum += float64(w) * powerSpec[filter.startBin+i]
			}
			feats[m][f] = float32(math.Log(sum + logGuard))
		}
	}

	return feats
}

// readPaddedPreemph virtually maps a padded array index back to the original audio,
// computing PyTorch-style reflect padding and pre-emphasis completely on-the-fly.
func readPaddedPreemph(samples []float32, paddedIdx, padAmt int, preemph float32) float32 {
	n := len(samples)
	if n == 0 {
		return 0.0
	}
	if n == 1 {
		return samples[0] // Pre-emph on 1 sample is just x[0]
	}

	origIdx := paddedIdx - padAmt

	// PyTorch pad_mode='reflect' exact logic
	if origIdx < 0 {
		origIdx = -origIdx
	} else if origIdx >= n {
		overshoot := origIdx - (n - 1)
		origIdx = (n - 1) - overshoot
	}

	// Foolproof bounded clamp to prevent panics on extremely tiny edge signals
	origIdx = max(0, min(origIdx, n-1))

	val := samples[origIdx]
	if preemph != 0.0 && origIdx > 0 {
		val -= preemph * samples[origIdx-1]
	}
	return val
}

func hannWindow(n int) []float32 {
	w := make([]float32, n)
	factor := 2.0 * math.Pi / float64(n)
	for i := range n {
		w[i] = float32(0.5 - 0.5*math.Cos(float64(i)*factor))
	}
	return w
}

func melFilterbank(nMels, nFFT, sampleRate int) [][]float32 {
	nBins := nFFT/2 + 1
	fMin, fMax := 0.0, float64(sampleRate)/2.0

	fftFreqs := make([]float64, nBins)
	for k := range nBins {
		fftFreqs[k] = float64(k) * fMax / float64(nBins-1)
	}

	melMin := hzToMelSlaney(fMin)
	melMax := hzToMelSlaney(fMax)
	hzPoints := make([]float64, nMels+2)
	for i := range nMels + 2 {
		mel := melMin + (melMax-melMin)*float64(i)/float64(nMels+1)
		hzPoints[i] = melToHzSlaney(mel)
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
			row[k] = float32(max(0.0, math.Min(lower, upper)))
		}
		enorm := 2.0 / (hzPoints[m+2] - hzPoints[m])
		for k := range row {
			row[k] = float32(float64(row[k]) * enorm)
		}
		bank[m] = row
	}
	return bank
}

func hzToMelSlaney(hz float64) float64 {
	const fMin = 0.0
	const fSp = 200.0 / 3.0
	const minLogHz = 1000.0
	minLogMel := (minLogHz - fMin) / fSp
	logstep := math.Log(6.4) / 27.0
	if hz < minLogHz {
		return (hz - fMin) / fSp
	}
	return minLogMel + math.Log(hz/minLogHz)/logstep
}

func melToHzSlaney(mel float64) float64 {
	const fMin = 0.0
	const fSp = 200.0 / 3.0
	const minLogHz = 1000.0
	minLogMel := (minLogHz - fMin) / fSp
	logstep := math.Log(6.4) / 27.0
	if mel < minLogMel {
		return fMin + fSp*mel
	}
	return minLogHz * math.Exp(logstep*(mel-minLogMel))
}

// fftPlan holds precomputed twiddle factors and bit-reversal indices
// for an extremely fast, zero-allocation Radix-2 Cooley-Tukey FFT.
type fftPlan struct {
	n       int
	bitRev  []int
	twiddle []complex128
}

func newFFTPlan(n int) *fftPlan {
	plan := &fftPlan{
		n:       n,
		bitRev:  make([]int, n),
		twiddle: make([]complex128, n/2),
	}

	// Precompute bit reversal to avoid loops and branching during inference
	for i := range n {
		j := 0
		tmp := i
		bit := n >> 1
		for bit > 0 {
			if tmp&1 == 1 {
				j |= bit
			}
			tmp >>= 1
			bit >>= 1
		}
		plan.bitRev[i] = j
	}

	// Precompute twiddle factors (W_N^k)
	for i := range n / 2 {
		ang := -2.0 * math.Pi * float64(i) / float64(n)
		plan.twiddle[i] = complex(math.Cos(ang), math.Sin(ang))
	}

	return plan
}

func (p *fftPlan) exec(x []complex128) {
	n := p.n
	if n <= 1 {
		return
	}

	for i := range n {
		j := p.bitRev[i]
		if i < j {
			x[i], x[j] = x[j], x[i]
		}
	}

	for length := 2; length <= n; length <<= 1 {
		half := length >> 1
		step := n / length
		for i := 0; i < n; i += length {
			for j := 0; j < half; j++ {
				w := p.twiddle[j*step]
				u := x[i+j]
				v := x[i+j+half] * w
				x[i+j] = u + v
				x[i+j+half] = u - v
			}
		}
	}
}
