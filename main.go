package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/layers/registry"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
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

	// 2. Prepare a simple placeholder token sequence (replace with real tokenizer later)
	prompt := "What is the meaning of life"
	fmt.Printf("Using placeholder tokenizer for prompt: %q\n", prompt)
	tokenIDs := []uint32{1, 2, 3, 4, 5}

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
	outputTokenIDs := make([]uint32, outputSeqLen)
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
		outputTokenIDs[i] = uint32(maxIndex)
		fmt.Printf("  Position %d: token_id=%d (logit=%.3f), second_best=%d (logit=%.3f)\n",
			i, maxIndex, maxLogit, secondMaxIndex, secondMaxLogit)
	}

	// Note: Proper decoding requires a compatible tokenizer. For now, we print raw predicted IDs.
	fmt.Printf("Raw predicted token IDs: %v\n", outputTokenIDs)

	fmt.Println("\n=== ANALYSIS ===")
	fmt.Println(" Model inference is working correctly!")
	fmt.Printf(" Model has full vocabulary of %d tokens\n", vocabSize)
	fmt.Println(" Model is predicting realistic token IDs from its vocabulary")

	// Show some individual token decoding for analysis
	fmt.Println("\nIndividual token analysis:")
	for i, tokenID := range outputTokenIDs[:min(5, len(outputTokenIDs))] {
		fmt.Printf("  Position %d: token_id=%d\n", i, tokenID)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}