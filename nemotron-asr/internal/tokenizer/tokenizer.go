// Package tokenizer provides a minimal reader for SentencePiece ".model"
// files. NeMo's ASR tokenizers only need id -> piece lookups at inference
// time (the RNN-T prediction network feeds back token ids, never re-tokenizes
// text), so this package deliberately implements only enough of the
// SentencePiece ModelProto wire format to pull out the ordered list of piece
// strings. It does not implement encoding, BPE merges, or scoring.
package tokenizer

import (
	"fmt"
	"os"
	"strings"
)

// Tokenizer holds the ordered vocabulary extracted from a SentencePiece
// model file. Token id N corresponds to Pieces[N].
type Tokenizer struct {
	Pieces []string
}

// Load reads and parses a SentencePiece .model file at path.
func Load(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading tokenizer file: %w", err)
	}
	pieces, err := parseModelProto(data)
	if err != nil {
		return nil, fmt.Errorf("parsing sentencepiece model: %w", err)
	}
	if len(pieces) == 0 {
		return nil, fmt.Errorf("no pieces found in sentencepiece model %s", path)
	}
	return &Tokenizer{Pieces: pieces}, nil
}

// IDToPiece returns the raw piece string for a token id, or "" if out of range.
func (t *Tokenizer) IDToPiece(id int) string {
	if id < 0 || id >= len(t.Pieces) {
		return ""
	}
	return t.Pieces[id]
}

// Detokenize joins a sequence of token ids into human-readable text,
// following the standard SentencePiece convention where "▁" (U+2581) marks
// the start of a word (i.e. is rendered as a space).
func (t *Tokenizer) Detokenize(ids []int) string {
	var sb strings.Builder
	for i, id := range ids {
		piece := t.IDToPiece(id)
		if piece == "" {
			continue
		}
		if strings.HasPrefix(piece, "\u2581") {
			if i != 0 {
				sb.WriteByte(' ')
			}
			piece = strings.TrimPrefix(piece, "\u2581")
		}
		sb.WriteString(piece)
	}
	return sb.String()
}

// --- minimal protobuf (proto2) wire-format decoding ---
//
// message ModelProto {
//   repeated SentencePiece pieces = 1;   // field 1, wire type 2 (LEN)
//   ...
// }
// message SentencePiece {
//   optional string piece = 1;           // field 1, wire type 2 (LEN)
//   optional float score = 2;
//   optional Type type = 3;
//   ...
// }
//
// We only need to walk top-level field 1 (pieces) and, within each, field 1
// (the piece string), skipping everything else per the wire type.

const (
	wireVarint  = 0
	wireFixed64 = 1
	wireLen     = 2
	wireFixed32 = 5
)

func parseModelProto(data []byte) ([]string, error) {
	var pieces []string
	pos := 0
	for pos < len(data) {
		fieldNum, wireType, n, err := readTag(data[pos:])
		if err != nil {
			return nil, err
		}
		pos += n
		switch wireType {
		case wireLen:
			length, n, err := readVarint(data[pos:])
			if err != nil {
				return nil, err
			}
			pos += n
			end := pos + int(length)
			if end > len(data) {
				return nil, fmt.Errorf("truncated message")
			}
			payload := data[pos:end]
			pos = end
			if fieldNum == 1 {
				piece, err := parseSentencePiece(payload)
				if err != nil {
					return nil, err
				}
				pieces = append(pieces, piece)
			}
		case wireVarint:
			_, n, err := readVarint(data[pos:])
			if err != nil {
				return nil, err
			}
			pos += n
		case wireFixed64:
			pos += 8
		case wireFixed32:
			pos += 4
		default:
			return nil, fmt.Errorf("unsupported wire type %d", wireType)
		}
	}
	return pieces, nil
}

// parseSentencePiece extracts the "piece" (field 1) string from a single
// embedded SentencePiece message, skipping other fields (score, type, etc).
func parseSentencePiece(data []byte) (string, error) {
	pos := 0
	piece := ""
	for pos < len(data) {
		fieldNum, wireType, n, err := readTag(data[pos:])
		if err != nil {
			return "", err
		}
		pos += n
		switch wireType {
		case wireLen:
			length, n, err := readVarint(data[pos:])
			if err != nil {
				return "", err
			}
			pos += n
			end := pos + int(length)
			if end > len(data) {
				return "", fmt.Errorf("truncated sentencepiece message")
			}
			if fieldNum == 1 {
				piece = string(data[pos:end])
			}
			pos = end
		case wireVarint:
			_, n, err := readVarint(data[pos:])
			if err != nil {
				return "", err
			}
			pos += n
		case wireFixed64:
			pos += 8
		case wireFixed32:
			pos += 4
		default:
			return "", fmt.Errorf("unsupported wire type %d", wireType)
		}
	}
	return piece, nil
}

func readTag(data []byte) (fieldNum int, wireType int, n int, err error) {
	v, n, err := readVarint(data)
	if err != nil {
		return 0, 0, 0, err
	}
	return int(v >> 3), int(v & 0x7), n, nil
}

func readVarint(data []byte) (uint64, int, error) {
	var result uint64
	var shift uint
	for i := 0; i < len(data); i++ {
		b := data[i]
		result |= uint64(b&0x7F) << shift
		if b&0x80 == 0 {
			return result, i + 1, nil
		}
		shift += 7
		if shift >= 64 {
			return 0, 0, fmt.Errorf("varint too long")
		}
	}
	return 0, 0, fmt.Errorf("truncated varint")
}
