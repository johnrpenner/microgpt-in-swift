# MicroGPT - Swift Edition

A minimal GPT implementation in pure Swift with no dependencies, based on Andrej Karpathy's micrograd GPT.

## Compilation

```bash
swiftc -o microgpt microgpt.swift
```

This will create an executable called `microgpt`.

## Usage

### Training Mode

Train the model on one or more text files:

```bash
./microgpt --train file1.txt file2.txt file3.txt --steps 1000 --model mymodel.txt
```

**Options:**
- `--train` - Enter training mode
- `--steps N` - Number of training steps (default: 1000)
- `--model path` - Path to save the model (default: model.txt)

### Generation Mode

Generate text using a trained model:

```bash
./microgpt --generate --model mymodel.txt --prompt "hello" --temp 0.7
```

**Options:**
- `--generate` - Enter generation mode
- `--model path` - Path to load the model from (default: model.txt)
- `--prompt "text"` - Starting prompt for generation
- `--temp N` - Temperature for sampling (0.1 = conservative, 1.0 = creative)

## Training Tips

### 1. Prepare Your Training Data

Create text files with the content you want to train on. The model works best with:

**For names/words:**
```
alice
bob
charlie
david
emma
```

**For sentences:**
```
the quick brown fox jumps over the lazy dog
swift is a powerful programming language
machine learning is fascinating
```

**For code snippets:**
```
func hello() { print("world") }
let x = 42
var array = [1, 2, 3]
```

### 2. Training Steps

- **Small datasets (100-1000 lines):** 500-1000 steps
- **Medium datasets (1000-5000 lines):** 1000-2000 steps
- **Larger datasets:** 2000-5000 steps

More steps = better memorization, but diminishing returns after a point.

### 3. Model Size Considerations

The current model is intentionally tiny:
- 1 layer
- 4 attention heads
- 16-dimensional embeddings
- ~10,000 parameters

This means:
- ✅ Fast training (seconds to minutes)
- ✅ Can run on any Mac
- ✅ Great for learning and experimentation
- ⚠️ Limited capacity (won't learn complex patterns)
- ⚠️ Works best on small, focused datasets

### 4. What Works Well

**Good use cases:**
- Learning name patterns (train on names, generate new ones)
- Simple text patterns (email addresses, URLs)
- Short code snippets
- Poetry/haikus (character-level patterns)
- Simple language patterns

**Less effective:**
- Long-form text generation
- Complex reasoning
- Multi-paragraph coherence
- Large vocabularies

### 5. Temperature Settings

When generating:
- `--temp 0.1` - Very conservative, repetitive
- `--temp 0.5` - Balanced (default)
- `--temp 0.8` - More creative
- `--temp 1.0` - Very creative, possibly nonsensical

## Example Workflows

### Example 1: Name Generator

Create a file `names.txt`:
```
alexander
benjamin
charlotte
daniel
elizabeth
```

Train:
```bash
./microgpt --train names.txt --steps 1000 --model names_model.txt
```

Generate:
```bash
./microgpt --generate --model names_model.txt --temp 0.6
```

### Example 2: Code Patterns

Create `code.txt`:
```
func add(a: Int, b: Int) -> Int { return a + b }
func multiply(x: Int, y: Int) -> Int { return x * y }
let sum = add(a: 5, b: 3)
var total = 0
```

Train:
```bash
./microgpt --train code.txt --steps 2000 --model code_model.txt
```

Generate:
```bash
./microgpt --generate --model code_model.txt --temp 0.5
```

### Example 3: Multiple Files

```bash
./microgpt --train data/file1.txt data/file2.txt data/file3.txt --steps 1500
```

## Understanding the Output

During training, you'll see:
```
step  100 / 1000 | loss 2.3456
```

- **Loss** starts high (3-4) and should decrease
- Final loss around 0.5-1.5 is good for this model size
- If loss stays high (>3), try more training steps
- If loss is very low (<0.1), model may be overfitting

During generation, you'll see 10 samples:
```
sample  1: alexander
sample  2: benjami
sample  3: elizabe
```

These are completely new, "hallucinated" outputs based on learned patterns.

## Troubleshooting

**"Could not read file"**
- Check file paths are correct
- Ensure files are UTF-8 encoded text

**Loss not decreasing**
- Try more training steps
- Ensure training data has patterns to learn
- Check that files actually contain text

**Generates gibberish**
- Train for more steps
- Lower temperature (--temp 0.3)
- May need more/better training data

**Crashes or hangs**
- Files too large - try smaller batches
- Model runs out of memory - reduce dataset size

## Technical Notes

- **Character-level model:** Works on individual characters, not words
- **Autoregressive:** Generates one character at a time
- **Causal attention:** Only looks at previous characters
- **No GPU acceleration:** Pure Swift, runs on CPU
- **Deterministic training:** Same data + steps = same model

## Limitations

This is an educational implementation:
- Not optimized for speed
- No batch processing
- No GPU support
- Very small model capacity
- Not suitable for production use

Perfect for understanding how GPT models work under the hood!

## License

Based on Andrej Karpathy's work. For educational purposes.
