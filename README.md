# judge-calibrate

Measure and correct position, length, self-preference and scale biases in LLM-as-judge scores.

## Install

```
pip install -e .
```

## Usage

```
judge-calibrate synth --n 200 --out data.jsonl
judge-calibrate probe data.jsonl --fake position_bias=0.8 --fake self_bias=0.5 --family claude --out report.json
judge-calibrate calibrate data.jsonl --method isotonic --length-control --save model.json
```
