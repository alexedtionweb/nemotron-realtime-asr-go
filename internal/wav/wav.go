// Package wav implements just enough of the RIFF/WAVE format to load PCM
// audio for ASR: 16-bit integer or 32-bit float, mono or multi-channel,
// at any sample rate. ReadFile always returns mono float32 samples in
// [-1, 1], resampled to targetSampleRate using a band-limited (windowed-sinc)
// resampler, which anti-alias-filters when downsampling.
package wav

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// ReadFile loads a WAV file and returns mono float32 samples resampled to
// targetSampleRate.
func ReadFile(path string, targetSampleRate int) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading wav file: %w", err)
	}
	samples, channels, sampleRate, err := decode(data)
	if err != nil {
		return nil, err
	}
	mono := toMono(samples, channels)
	if sampleRate != targetSampleRate {
		mono = resampleBandlimited(mono, sampleRate, targetSampleRate)
	}
	return mono, nil
}

// decode parses the RIFF/WAVE container and returns interleaved float32
// samples in [-1, 1], the channel count, and the sample rate.
func decode(data []byte) (samples []float32, channels int, sampleRate int, err error) {
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, 0, fmt.Errorf("not a RIFF/WAVE file")
	}
	pos := 12
	var (
		audioFormat   uint16
		bitsPerSample uint16
		haveFmt       bool
		pcmData       []byte
		haveData      bool
	)
	for pos+8 <= len(data) {
		chunkID := string(data[pos : pos+4])
		chunkSize := int(binary.LittleEndian.Uint32(data[pos+4 : pos+8]))
		bodyStart := pos + 8
		bodyEnd := bodyStart + chunkSize
		if bodyEnd > len(data) {
			bodyEnd = len(data)
		}
		switch chunkID {
		case "fmt ":
			if bodyEnd-bodyStart < 16 {
				return nil, 0, 0, fmt.Errorf("fmt chunk too short")
			}
			body := data[bodyStart:bodyEnd]
			audioFormat = binary.LittleEndian.Uint16(body[0:2])
			channels = int(binary.LittleEndian.Uint16(body[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(body[4:8]))
			bitsPerSample = binary.LittleEndian.Uint16(body[14:16])
			haveFmt = true
		case "data":
			pcmData = data[bodyStart:bodyEnd]
			haveData = true
		}
		pos = bodyEnd
		if chunkSize%2 == 1 {
			pos++ // chunks are word-aligned
		}
	}
	if !haveFmt || !haveData {
		return nil, 0, 0, fmt.Errorf("missing fmt or data chunk")
	}

	const formatPCM = 1
	const formatIEEEFloat = 3
	// WAVE_FORMAT_EXTENSIBLE (0xFFFE) is common from modern recorders; treat
	// it like PCM/float based on bitsPerSample since the true sub-format is
	// in bytes we don't bother parsing.
	switch audioFormat {
	case formatPCM, 0xFFFE:
		samples, err = decodePCM(pcmData, int(bitsPerSample))
	case formatIEEEFloat:
		samples, err = decodeFloat(pcmData, int(bitsPerSample))
	default:
		return nil, 0, 0, fmt.Errorf("unsupported wav audio format code %d", audioFormat)
	}
	if err != nil {
		return nil, 0, 0, err
	}
	return samples, channels, sampleRate, nil
}

func decodePCM(data []byte, bitsPerSample int) ([]float32, error) {
	switch bitsPerSample {
	case 16:
		n := len(data) / 2
		out := make([]float32, n)
		for i := range n {
			v := int16(binary.LittleEndian.Uint16(data[i*2 : i*2+2]))
			out[i] = float32(v) / 32768.0
		}
		return out, nil
	case 32:
		n := len(data) / 4
		out := make([]float32, n)
		for i := range n {
			v := int32(binary.LittleEndian.Uint32(data[i*4 : i*4+4]))
			out[i] = float32(v) / 2147483648.0
		}
		return out, nil
	case 8:
		n := len(data)
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = (float32(data[i]) - 128) / 128.0
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported PCM bit depth %d", bitsPerSample)
	}
}

func decodeFloat(data []byte, bitsPerSample int) ([]float32, error) {
	if bitsPerSample != 32 {
		return nil, fmt.Errorf("unsupported float bit depth %d", bitsPerSample)
	}
	n := len(data) / 4
	out := make([]float32, n)
	for i := range n {
		bits := binary.LittleEndian.Uint32(data[i*4 : i*4+4])
		out[i] = math.Float32frombits(bits)
	}
	return out, nil
}

// toMono downmixes interleaved multi-channel samples by averaging channels.
func toMono(samples []float32, channels int) []float32 {
	if channels <= 1 {
		return samples
	}
	n := len(samples) / channels
	out := make([]float32, n)
	for i := range n {
		var sum float32
		for c := range channels {
			sum += samples[i*channels+c]
		}
		out[i] = sum / float32(channels)
	}
	return out
}

// resampleBandlimited performs windowed-sinc ("band-limited") resampling:
// for downsampling this bakes in an anti-aliasing low-pass filter at the
// destination Nyquist frequency, unlike naive linear interpolation (which
// lets frequency content above the new Nyquist alias back into the
// passband and corrupt the signal — exactly the kind of distortion that
// degrades ASR accuracy). Uses a Blackman-windowed sinc kernel with 16
// zero-crossings on each side; slower than linear interpolation but still
// trivially fast for offline/chunk-sized audio.
func resampleBandlimited(in []float32, srcRate, dstRate int) []float32 {
	if srcRate == dstRate || len(in) == 0 {
		return in
	}
	ratio := float64(dstRate) / float64(srcRate)
	cutoff := ratio
	if cutoff > 1.0 {
		cutoff = 1.0 // upsampling: sinc interpolation is already band-limited to the source Nyquist
	}
	const zeroCrossings = 16.0
	halfWidth := zeroCrossings / cutoff // in source samples

	outLen := int(float64(len(in)) * ratio)
	out := make([]float32, outLen)

	for i := range outLen {
		srcPos := float64(i) / ratio
		nStart := int(math.Floor(srcPos - halfWidth))
		nEnd := int(math.Ceil(srcPos + halfWidth))
		var sum float64
		for n := nStart; n <= nEnd; n++ {
			if n < 0 || n >= len(in) {
				continue
			}
			d := srcPos - float64(n)
			var tap float64
			if math.Abs(d) < 1e-9 {
				tap = cutoff
			} else {
				x := cutoff * d
				sincVal := math.Sin(math.Pi*x) / (math.Pi * x)
				// Blackman window over the support [-halfWidth, halfWidth].
				t := (d + halfWidth) / (2 * halfWidth) // -> [0, 1]
				window := 0.42 - 0.5*math.Cos(2*math.Pi*t) + 0.08*math.Cos(4*math.Pi*t)
				tap = cutoff * sincVal * window
			}
			sum += float64(in[n]) * tap
		}
		out[i] = float32(sum)
	}
	return out
}
