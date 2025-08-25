package main

import (
	"fmt"
	"log"

	"github.com/sugarme/tokenizer/pretrained"
)

// debugTokenizer is a helper to manually test tokenizer behavior.
// Call this from tests or a separate cmd as needed.
func debugTokenizer() {
	tk, err := pretrained.FromFile("data/tokenizer.json")
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	prompt := "What is the meaning of life"
	encoding, err := tk.EncodeSingle(prompt)
	if err != nil {
		log.Fatalf("Failed to encode prompt: %v", err)
	}

	fmt.Printf("Encoded prompt '%s' -> %v\n", prompt, encoding.Ids)
}
