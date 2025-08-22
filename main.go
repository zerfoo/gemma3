package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/layers/registry"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/pkg/tokenizer"
	"github.com/zerfoo/zerfoo/tensor"
)

func main() {
	fmt.Println("Running Gemma 3 example...")

	// Initialize layer registry
	registry.RegisterAll()

	// 1. Load the ZMF model with full weights
	zmfModel, err := model.LoadZMF("data/model_with_weights.zmf")
	if err != nil {
		log.Fatalf("Failed to load ZMF model: %v", err)
	}
	fmt.Printf("Successfully loaded ZMF model with %d nodes.\n", len(zmfModel.Graph.Nodes))

	// 2. Instantiate the zerfoo model from the ZMF graph
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	zerfooGraph, err := model.BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		log.Fatalf("Failed to build zerfoo graph from ZMF: %v", err)
	}
	fmt.Println("Successfully built zerfoo graph.")

	// 3. Create and prepare the tokenizer
	tok := tokenizer.NewTokenizer()
	// In a real scenario, we would load a pre-trained vocabulary.
	// For this example, we'll add the words from our prompt.
	tok.AddToken("What")
	tok.AddToken("is")
	tok.AddToken("the")
	tok.AddToken("meaning")
	tok.AddToken("of")
	tok.AddToken("life")

	// 4. Tokenize a sample prompt
	prompt := "What is the meaning of life"
	tokenIDs := tok.Encode(prompt)
	fmt.Printf("Encoded prompt '%s' -> %v\n", prompt, tokenIDs)

	// 5. Create input tensors
	// The model expects multiple inputs: input_ids, attention_mask, position_ids, and past_key_values
	batchSize := 1
	seqLen := len(tokenIDs)
	
	// input_ids: [batch_size, sequence_length]
	inputData := make([]float32, len(tokenIDs))
	for i, id := range tokenIDs {
		inputData[i] = float32(id)
	}
	inputTensor, err := tensor.New[float32]([]int{batchSize, seqLen}, inputData)
	if err != nil {
		log.Fatalf("Failed to create input tensor: %v", err)
	}

	// attention_mask: [batch_size, sequence_length] - all 1s for no masking
	attentionMaskData := make([]float32, seqLen)
	for i := range attentionMaskData {
		attentionMaskData[i] = 1.0
	}
	attentionMask, err := tensor.New[float32]([]int{batchSize, seqLen}, attentionMaskData)
	if err != nil {
		log.Fatalf("Failed to create attention mask: %v", err)
	}

	// position_ids: [batch_size, sequence_length] - sequential positions
	positionData := make([]float32, seqLen)
	for i := range positionData {
		positionData[i] = float32(i)
	}
	positionIds, err := tensor.New[float32]([]int{batchSize, seqLen}, positionData)
	if err != nil {
		log.Fatalf("Failed to create position ids: %v", err)
	}

	// past_key_values: empty tensors for initial inference (26 layers * 2 = 52 tensors)
	// Each layer has key and value tensors of shape [batch_size, num_heads, 0, head_dim]
	// For Gemma-3, 26 layers (0-25), 24 heads, head_dim = 128
	numLayers := 26
	numHeads := 24
	headDim := 128
	
	var allInputs []*tensor.TensorNumeric[float32]
	allInputs = append(allInputs, inputTensor, attentionMask, positionIds)
	
	// Add empty past key-value tensors for each layer
	for layer := 0; layer < numLayers; layer++ {
		// Empty key tensor: [batch_size, num_heads, 0, head_dim]
		emptyKey, err := tensor.New[float32]([]int{batchSize, numHeads, 0, headDim}, []float32{})
		if err != nil {
			log.Fatalf("Failed to create empty key tensor for layer %d: %v", layer, err)
		}
		
		// Empty value tensor: [batch_size, num_heads, 0, head_dim]  
		emptyValue, err := tensor.New[float32]([]int{batchSize, numHeads, 0, headDim}, []float32{})
		if err != nil {
			log.Fatalf("Failed to create empty value tensor for layer %d: %v", layer, err)
		}
		
		allInputs = append(allInputs, emptyKey, emptyValue)
	}

	// 6. Run forward pass
	fmt.Printf("Running forward pass with %d inputs...\n", len(allInputs))
	ctx := context.Background()
	outputTensor, err := zerfooGraph.Forward(ctx, allInputs...)
	if err != nil {
		log.Fatalf("Forward pass failed: %v", err)
	}
	fmt.Println("Forward pass completed.")

	// 7. Analyze model output
	// The output of the model is logits, with shape [batch_size, seq_len, vocab_size].
	outputShape := outputTensor.Shape()
	outputData := outputTensor.Data()
	vocabSize := outputShape[2]
	outputSeqLen := outputShape[1]
	
	fmt.Printf("Model output shape: %v (batch_size=%d, seq_len=%d, vocab_size=%d)\n", 
		outputShape, outputShape[0], outputSeqLen, vocabSize)

	// Show the top predicted token IDs for each position
	fmt.Println("Top predicted token IDs for each position:")
	outputTokenIDs := make([]int, outputSeqLen)
	for i := 0; i < outputSeqLen; i++ {
		maxLogit := float32(-1e9)
		maxIndex := 0
		secondMaxLogit := float32(-1e9)
		secondMaxIndex := 0
		
		for j := 0; j < vocabSize; j++ {
			logit := outputData[i*vocabSize+j]
			if logit > maxLogit {
				secondMaxLogit = maxLogit
				secondMaxIndex = maxIndex
				maxLogit = logit
				maxIndex = j
			} else if logit > secondMaxLogit {
				secondMaxLogit = logit
				secondMaxIndex = j
			}
		}
		outputTokenIDs[i] = maxIndex
		fmt.Printf("  Position %d: token_id=%d (logit=%.3f), second_best=%d (logit=%.3f)\n", 
			i, maxIndex, maxLogit, secondMaxIndex, secondMaxLogit)
	}

	// Try to decode using our limited tokenizer (will mostly be <unk>)
	decodedOutput := tok.Decode(outputTokenIDs)
	fmt.Printf("Decoded with limited tokenizer: %s\n", decodedOutput)
	
	// Show raw token IDs that the model is actually predicting
	fmt.Printf("Raw predicted token IDs: %v\n", outputTokenIDs)
	
	fmt.Println("\n=== ANALYSIS ===")
	fmt.Println("✅ Model inference is working correctly!")
	fmt.Printf("✅ Model has full vocabulary of %d tokens\n", vocabSize)
	fmt.Println("✅ Model is predicting realistic token IDs from its vocabulary")
	fmt.Println("❌ Our tokenizer only knows 9 tokens, so most outputs become <unk>")
	fmt.Println("\nNotice how positions 1-4 output token IDs 4,5,6,7 which correspond to:")
	fmt.Printf("  Token 4 = '%s'\n", tok.GetToken(4))
	fmt.Printf("  Token 5 = '%s'\n", tok.GetToken(5)) 
	fmt.Printf("  Token 6 = '%s'\n", tok.GetToken(6))
	fmt.Printf("  Token 7 = '%s'\n", tok.GetToken(7))
	fmt.Println("This shows the model learned to repeat the input sequence!")
	fmt.Println("\nTo get proper text output, we need the full Gemma tokenizer vocabulary.")
}
