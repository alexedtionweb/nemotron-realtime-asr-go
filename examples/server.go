// Command server uses the nemotron SDK to expose a real-time speech-to-text
// websocket endpoint for browsers.
package main

import (
	_ "embed"
	"encoding/json"
	"flag"
	"log"
	"net/http"

	"github.com/alexedtionweb/nemotron-realtime-asr-go"
	"github.com/gorilla/websocket"
)

//go:embed index.html
var indexHTML []byte

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
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

func main() {
	modelDir := flag.String("model-dir", "/path/to/nemotron/model", "Path to the downloaded ONNX model directory")
	ortLib := flag.String("ort-lib", "/usr/local/lib/libonnxruntime.so", "Path to the ONNX Runtime shared library")
	listenAddr := flag.String("addr", ":8081", "HTTP server address to listen on")
	flag.Parse()

	// 1. Initialize the ASR Engine using the SDK
	engine, err := nemotron.NewEngine(*modelDir, *ortLib)
	if err != nil {
		log.Fatalf("Failed to initialize Nemotron engine: %v", err)
	}
	defer engine.Close()

	// 2. Setup standard HTTP Routes
	mux := http.NewServeMux()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexHTML)
	})

	mux.HandleFunc("/languages", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(engine.AvailableLanguages())
	})

	mux.HandleFunc("/ws/transcribe", func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("WebSocket upgrade failed: %v", err)
			return
		}
		handleTranscription(conn, engine)
	})

	log.Printf("Server listening on http://localhost%s", *listenAddr)
	if err := http.ListenAndServe(*listenAddr, mux); err != nil {
		log.Fatal(err)
	}
}

// handleTranscription runs the websocket loop, utilizing an SDK Session.
func handleTranscription(conn *websocket.Conn, engine *nemotron.Engine) {
	defer conn.Close()
	log.Printf("Microphone client connected: %s", conn.RemoteAddr())

	var session *nemotron.Session

	// Ensure session is closed and model returned to pool if the websocket drops
	defer func() {
		if session != nil {
			session.Close()
		}
	}()

	for {
		messageType, payload, err := conn.ReadMessage()
		if err != nil {
			return
		}

		switch messageType {
		case websocket.TextMessage:
			var msg clientMessage
			if err := json.Unmarshal(payload, &msg); err != nil {
				writeMessage(conn, transcriptMessage{Type: "error", Error: "invalid control message"})
				continue
			}

			switch msg.Type {
			case "start":
				if session != nil {
					session.Close()
				}

				language := msg.Language
				if language == "" {
					language = "auto"
				}

				session, err = engine.NewSession(language)
				if err != nil {
					writeMessage(conn, transcriptMessage{Type: "error", Error: err.Error()})
					session = nil
					continue
				}
				writeMessage(conn, transcriptMessage{Type: "ready"})

			case "end":
				if session != nil {
					finalText, err := session.Finalize()
					if err != nil {
						writeMessage(conn, transcriptMessage{Type: "error", Error: err.Error()})
					}
					writeMessage(conn, transcriptMessage{Type: "final", Text: finalText})
					session.Close()
					session = nil
				}
			}

		case websocket.BinaryMessage:
			if session == nil {
				writeMessage(conn, transcriptMessage{Type: "error", Error: "send a start message before audio"})
				continue
			}

			// The SDK handles all float conversion, buffering, and chunk sizes internally.
			text, err := session.WritePCM16(payload)
			if err != nil {
				writeMessage(conn, transcriptMessage{Type: "error", Error: err.Error()})
				continue
			}

			writeMessage(conn, transcriptMessage{Type: "partial", Text: text})
		}
	}
}

func writeMessage(conn *websocket.Conn, message transcriptMessage) {
	b, _ := json.Marshal(message)
	if err := conn.WriteMessage(websocket.TextMessage, b); err != nil {
		log.Printf("write WebSocket message: %v", err)
	}
}
