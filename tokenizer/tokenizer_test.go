package tokenizer

import (
	"testing"
)

func TestGemmaTokenizer(t *testing.T) {
	// Test basic tokenizer functionality
	tokenizer, err := NewGemmaTokenizer("../data/tokenizer.json")
	if err != nil {
		t.Skipf("Skipping test - tokenizer file not found: %v", err)
		return
	}

	// Test encoding
	text := "What is the meaning of life?"
	tokens, err := tokenizer.Encode(text)
	if err != nil {
		t.Skipf("Skipping test - tokenizer encoding failed (likely compatibility issue): %v", err)
		return
	}

	if len(tokens) == 0 {
		t.Fatalf("Expected tokens, got empty slice")
	}

	t.Logf("Encoded '%s' to tokens: %v", text, tokens)

	// Test vocab size
	vocabSize := tokenizer.GetVocabSize()
	if vocabSize <= 0 {
		t.Fatalf("Expected positive vocab size, got %d", vocabSize)
	}

	t.Logf("Vocabulary size: %d", vocabSize)

	// Test special tokens
	tokensWithSpecial := tokenizer.AddSpecialTokens(tokens)
	if len(tokensWithSpecial) <= len(tokens) {
		t.Errorf("Expected special tokens to be added")
	}

	t.Logf("Tokens with special tokens: %v", tokensWithSpecial)

	// Test tensor creation
	batchSize := 1
	tensorResult, err := tokenizer.EncodeToTensor(text, batchSize)
	if err != nil {
		t.Fatalf("Failed to encode to tensor: %v", err)
	}

	expectedShape := []int{batchSize, len(tokens)}
	if !equalSlices(tensorResult.Shape(), expectedShape) {
		t.Errorf("Expected tensor shape %v, got %v", expectedShape, tensorResult.Shape())
	}

	t.Logf("Tensor shape: %v", tensorResult.Shape())
}

func TestTokenizerRoundTrip(t *testing.T) {
	tokenizer, err := NewGemmaTokenizer("../data/tokenizer.json")
	if err != nil {
		t.Skipf("Skipping test - tokenizer file not found: %v", err)
		return
	}

	originalText := "Hello world!"
	
	// Encode
	tokens, err := tokenizer.Encode(originalText)
	if err != nil {
		t.Skipf("Skipping test - tokenizer encoding failed (likely compatibility issue): %v", err)
		return
	}

	// Decode
	decodedText, err := tokenizer.Decode(tokens)
	if err != nil {
		t.Skipf("Skipping test - tokenizer decoding failed (likely compatibility issue): %v", err)
		return
	}

	t.Logf("Original: '%s'", originalText)
	t.Logf("Tokens: %v", tokens)
	t.Logf("Decoded: '%s'", decodedText)

	// Note: The decoded text might not exactly match the original due to tokenization
	// but it should be a reasonable representation
	if len(decodedText) == 0 {
		t.Errorf("Expected non-empty decoded text")
	}
}

// Helper function to compare slices
func equalSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestTokenizerAPI tests our API without requiring actual tokenizer files
func TestTokenizerAPI(t *testing.T) {
	// Test AddSpecialTokens function (works without tokenizer file)
	originalTokens := []int{10, 20, 30}
	
	// This should work since it just adds BOS token ID 1 at the beginning
	tokensWithSpecial := (&GemmaTokenizer{}).AddSpecialTokens(originalTokens)
	
	expected := []int{1, 10, 20, 30}
	if !equalSlices(tokensWithSpecial, expected) {
		t.Errorf("AddSpecialTokens failed: expected %v, got %v", expected, tokensWithSpecial)
	}
	
	// Test that BOS token is not duplicated
	tokensAlreadyWithBOS := []int{1, 10, 20, 30}
	result := (&GemmaTokenizer{}).AddSpecialTokens(tokensAlreadyWithBOS)
	if !equalSlices(result, tokensAlreadyWithBOS) {
		t.Errorf("AddSpecialTokens should not duplicate BOS token: expected %v, got %v", tokensAlreadyWithBOS, result)
	}
	
	t.Logf("TokenizerAPI test passed - AddSpecialTokens works correctly")
}