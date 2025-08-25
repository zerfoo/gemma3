# Gemma 3 - Zerfoo Implementation

This repository contains a Gemma 3 language model implementation using the Zerfoo ML framework. The implementation has been updated to use the new Zerfoo architecture with ZMF (Zerfoo Model Format) support.

## Features

- **Modern Zerfoo Architecture**: Uses the latest Zerfoo framework with generic tensor types and graph-based computation
- **ZMF Model Loading**: Supports loading models from ZMF format instead of legacy ONNX
- **Tokenizer Integration**: Includes SentencePiece tokenizer support via `github.com/sugarme/tokenizer`
- **Comprehensive Testing**: Unit tests for all components plus integration tests
- **CPU Engine Support**: Optimized for CPU inference using Zerfoo's CPU engine

## Architecture

### Core Components

1. **Model (`gemma/gemma.go`)**: Main model combining embedding, transformer stack, and LM head
2. **GemmaStack (`gemma/gemma_stack.go`)**: Multi-layer transformer implementation with local attention
3. **Tokenizer (`tokenizer/tokenizer.go`)**: SentencePiece tokenizer wrapper for text processing

### Key Features

- **Generic Tensor Types**: Uses `TensorNumeric[T]` for type-safe numeric operations
- **Local Attention**: Implements sliding window attention with configurable window size
- **Weight Sharing**: Shares weights between embedding and LM head layers
- **Flexible Architecture**: Configurable number of layers, heads, dimensions, etc.

## Usage

### Basic Inference

```go
package main

import (
    "context"
    "github.com/zerfoo/gemma/tokenizer"
    "github.com/zerfoo/zerfoo/compute"
    "github.com/zerfoo/zerfoo/layers/registry"
    "github.com/zerfoo/zerfoo/model"
    "github.com/zerfoo/zerfoo/numeric"
)

func main() {
    // Initialize layer registry
    registry.RegisterAll()

    // Load ZMF model
    zmfModel, err := model.LoadZMF("data/model_with_weights.zmf")
    if err != nil {
        panic(err)
    }

    // Build Zerfoo graph
    ops := numeric.Float32Ops{}
    engine := compute.NewCPUEngine[float32](ops)
    graph, err := model.BuildFromZMF[float32](engine, ops, zmfModel)
    if err != nil {
        panic(err)
    }

    // Initialize tokenizer
    tokenizer, err := tokenizer.NewGemmaTokenizer("data/tokenizer.json")
    if err != nil {
        panic(err)
    }

    // Tokenize input
    tokens, err := tokenizer.Encode("What is the meaning of life?")
    if err != nil {
        panic(err)
    }

    // Run inference
    output, err := graph.Forward(context.Background(), /* inputs */)
    if err != nil {
        panic(err)
    }
}
```

### Model Creation (Programmatic)

```go
import "github.com/zerfoo/gemma/gemma"

// Create a Gemma model programmatically
model, err := gemma.New[float32](
    engine,           // Compute engine
    ops,              // Numeric operations
    vocabSize,        // Vocabulary size
    hiddenSize,       // Hidden dimension
    numHeads,         // Number of attention heads
    numKeyValueHeads, // Number of key-value heads
    ffnDim,           // Feed-forward dimension
    epsilon,          // RMS norm epsilon
    base,             // Rotary embedding base
    maxSeqLen,        // Maximum sequence length
    numLayers,        // Number of layers
    localWindowSize,  // Local attention window size
    globalInterval,   // Global attention interval
)
```

## Testing

The project includes comprehensive tests:

### Unit Tests
```bash
# Test model components
go test ./gemma/...

# Test tokenizer
go test ./tokenizer/...
```

### Integration Tests
```bash
# API integration (fast)
go test -run TestAPIIntegration

# Model architecture compatibility
go test -run TestModelArchitectureCompatibility

# End-to-end test (requires model files)
go test -run TestGemma3EndToEnd
```

### Test Coverage

- **Gemma Stack**: Tests forward pass with mock data
- **Gemma Model**: Tests full model forward pass
- **Tokenizer API**: Tests special token handling and encoding
- **API Integration**: Tests component compatibility and tensor operations
- **Architecture Tests**: Validates model creation with various parameters

## Model Files

The implementation expects these files in the `data/` directory:

- `model_with_weights.zmf`: ZMF model file with full weights
- `tokenizer.json`: SentencePiece tokenizer configuration
- Additional config files (optional)

## Dependencies

```go
require (
    github.com/sugarme/tokenizer v0.2.2
    github.com/zerfoo/zerfoo v0.2.0
)
```

## Performance

- **CPU Optimized**: Uses Zerfoo's optimized CPU engine
- **Memory Efficient**: Supports various quantization formats
- **Scalable**: Configurable model size and attention patterns

## Known Issues

1. **Tokenizer Compatibility**: Some Gemma tokenizer.json files may not be fully compatible with the sugarme/tokenizer library
2. **Large Model Loading**: Full-size models may require significant memory and loading time
3. **Test Timeouts**: Integration tests with large models may timeout

## Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass before submitting PRs

## License

See LICENSE file for details.