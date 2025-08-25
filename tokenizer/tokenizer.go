package tokenizer

import (
	"fmt"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	"github.com/zerfoo/zerfoo/tensor"
)

// GemmaTokenizer wraps the tokenizer functionality for Gemma models.
type GemmaTokenizer struct {
	tokenizer *tokenizer.Tokenizer
}

// NewGemmaTokenizer creates a new GemmaTokenizer from a tokenizer.json file.
func NewGemmaTokenizer(tokenizerPath string) (*GemmaTokenizer, error) {
	tk, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer from %s: %w", tokenizerPath, err)
	}

	return &GemmaTokenizer{
		tokenizer: tk,
	}, nil
}

// Encode converts a text string to token IDs.
func (gt *GemmaTokenizer) Encode(text string) ([]int, error) {
	encoding, err := gt.tokenizer.EncodeSingle(text, false) // don't add special tokens by default
	if err != nil {
		return nil, fmt.Errorf("failed to encode text: %w", err)
	}

	// Convert uint32 to int
	tokens := make([]int, len(encoding.Ids))
	for i, id := range encoding.Ids {
		tokens[i] = int(id)
	}

	return tokens, nil
}

// EncodeToTensor converts a text string to a tensor of token IDs.
func (gt *GemmaTokenizer) EncodeToTensor(text string, batchSize int) (*tensor.TensorNumeric[int], error) {
	tokens, err := gt.Encode(text)
	if err != nil {
		return nil, err
	}

	seqLen := len(tokens)
	
	// Create tensor with shape [batchSize, seqLen]
	tensorData := make([]int, batchSize*seqLen)
	for batch := 0; batch < batchSize; batch++ {
		copy(tensorData[batch*seqLen:(batch+1)*seqLen], tokens)
	}

	return tensor.New[int]([]int{batchSize, seqLen}, tensorData)
}

// Decode converts token IDs back to text.
func (gt *GemmaTokenizer) Decode(tokenIDs []int) (string, error) {
	text := gt.tokenizer.Decode(tokenIDs, true) // skip special tokens
	return text, nil
}

// GetVocabSize returns the vocabulary size of the tokenizer.
func (gt *GemmaTokenizer) GetVocabSize() int {
	return gt.tokenizer.GetVocabSize(true) // with added tokens
}

// AddSpecialTokens adds special tokens to the input sequence if needed.
func (gt *GemmaTokenizer) AddSpecialTokens(tokens []int) []int {
	// For Gemma models, we typically need to add BOS (Beginning of Sequence) token
	// The BOS token is usually token ID 1 for Gemma models
	bosToken := 1
	
	// Check if BOS token is already present
	if len(tokens) > 0 && tokens[0] == bosToken {
		return tokens
	}
	
	// Add BOS token at the beginning
	result := make([]int, len(tokens)+1)
	result[0] = bosToken
	copy(result[1:], tokens)
	
	return result
}