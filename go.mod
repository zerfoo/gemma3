module github.com/zerfoo/gemma3

go 1.25

require (
	github.com/sugarme/tokenizer v0.2.2
	github.com/zerfoo/zerfoo v0.2.0
	github.com/zerfoo/zonnx v0.0.0-20250822052139-1cca2c1c27db
	google.golang.org/protobuf v1.36.8
)

replace github.com/zerfoo/zonnx => ../zonnx

replace github.com/zerfoo/zmf => ../zmf

require (
	github.com/emirpasic/gods v1.12.0 // indirect
	github.com/mitchellh/colorstring v0.0.0-20190213212951-d06e56a500db // indirect
	github.com/rivo/uniseg v0.1.0 // indirect
	github.com/schollz/progressbar/v2 v2.15.0 // indirect
	github.com/sugarme/regexpset v0.0.0-20200920021344-4d4ec8eaf93c // indirect
	github.com/zerfoo/float16 v0.1.0 // indirect
	github.com/zerfoo/float8 v0.1.1 // indirect
	github.com/zerfoo/zmf v0.2.0 // indirect
	golang.org/x/text v0.15.0 // indirect
)
