package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/compute/cpu"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/pkg/tokenizer"
	"github.com/zerfoo/zerfoo/tensor"
)

func main() {
	fmt.Println("Running Gemma 3 example...")

	// 1. Load the ZMF model
	zmfModel, err := model.LoadZMF("data/model.zmf")
	if err != nil {
		log.Fatalf("Failed to load ZMF model: %v", err)
	}
	fmt.Printf("Successfully loaded ZMF model with %d nodes.\n", len(zmfModel.Graph.Nodes))

	// 2. Instantiate the zerfoo model from the ZMF graph
	engine := cpu.NewEngine[float32]()
	ops := numeric.NewFloat32Ops()
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

	// 5. Create input tensor
	// The model expects a 2D tensor of shape [batch_size, sequence_length]
	inputData := make([]float32, len(tokenIDs))
	for i, id := range tokenIDs {
		inputData[i] = float32(id)
	}
	inputTensor, err := tensor.NewTensor(inputData, []int{1, len(tokenIDs)})
	if err != nil {
		log.Fatalf("Failed to create input tensor: %v", err)
	}

	// 6. Run forward pass
	fmt.Println("Running forward pass...")
	ctx := context.Background()
	outputTensor, err := zerfooGraph.Forward(ctx, inputTensor)
	if err != nil {
		log.Fatalf("Forward pass failed: %v", err)
	}
	fmt.Println("Forward pass completed.")

	// 7. De-tokenize and print output
	// The output of the model is logits, with shape [batch_size, seq_len, vocab_size].
	// For this example, we'll just take the argmax along the vocab_size dimension
	// to get the most likely next token ID for each position.
	outputShape := outputTensor.Shape()
	outputData := outputTensor.Data()
	vocabSize := outputShape[2]
	seqLen := outputShape[1]

	outputTokenIDs := make([]int, seqLen)
	for i := 0; i < seqLen; i++ {
		maxLogit := float32(-1e9)
		maxIndex := 0
		for j := 0; j < vocabSize; j++ {
			logit := outputData[i*vocabSize+j]
			if logit > maxLogit {
				maxLogit = logit
				maxIndex = j
			}
		}
		outputTokenIDs[i] = maxIndex
	}

	decodedOutput := tok.Decode(outputTokenIDs)
	fmt.Printf("Model output: %s\n", decodedOutput)
}
