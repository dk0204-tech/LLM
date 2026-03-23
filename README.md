# DK — Language Model from Scratch

A decoder-only transformer language model built entirely from scratch in PyTorch. No wrapper libraries. Every component is written explicitly — tokenizer, attention, feed-forward network, training loop, and generation.

**Author:** Puneeth Ram  
**Architecture:** Decoder-only transformer (GPT-style)  
**Training objective:** Causal language modeling (next-token prediction)

---

## What This Is

DK is a research implementation of a modern language model, designed to be read and understood as much as it is run. The architecture follows conventions established by LLaMA and Mistral: RoPE positional encodings, RMSNorm, SwiGLU activation, and pre-norm placement throughout. Each design choice is explained inline in the notebook.

The default configuration targets roughly 10 million parameters — small enough to train on a laptop GPU in under an hour, large enough to produce coherent text on a simple corpus.

---

## Architecture

| Component         | Choice       | Why                                                                 |
|-------------------|--------------|---------------------------------------------------------------------|
| Position encoding | RoPE (Rotary)| Relative distances, better length generalization                   |
| Normalization     | RMSNorm      | Same quality as LayerNorm, ~10% faster                             |
| Norm placement    | Pre-norm     | More stable training at depth                                      |
| FFN activation    | SwiGLU       | More expressive than ReLU at equal parameter count                 |
| Biases            | None         | No measurable quality gain, removed for efficiency                 |
| Weight tying      | Embedding = LM head | Halves that matrix's parameters, improves perplexity          |

---

## Default Configuration

```python
d_model      = 384       # embedding dimension
n_layers     = 6         # transformer blocks
n_heads      = 6         # attention heads
head_dim     = 64        # d_model // n_heads
context_len  = 256       # tokens per sample
dropout      = 0.1

# Training
batch_size   = 64
grad_accum   = 4         # effective batch = 256
max_iters    = 5000
lr           = 3e-4      # cosine decay with 200-step warmup
weight_decay = 0.1
grad_clip    = 1.0
```

Estimated parameters: ~10M (non-embedding).

---

## Project Structure

```
DK_LLM.ipynb          Main notebook — all phases in sequence
dk_checkpoint.pt      Best checkpoint saved during training
dk_training_curves.png  Loss and perplexity plots
data/
    input.txt         Training corpus (Tiny Shakespeare or your own)
README.md             This file
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install tiktoken datasets matplotlib tqdm
```

### 2. Get training data

```bash
mkdir -p data
wget -q -O data/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Or use any plain text file. Point `cfg.data_path` at it.

### 3. Run the notebook

Open `DK_LLM.ipynb` and run all cells from top to bottom. Training starts automatically after the optimizer is configured.

---

## Training

The training loop uses:

- **Gradient accumulation** — simulates larger batches on limited VRAM
- **Mixed precision (bfloat16)** — halves memory, speeds up forward pass on Ampere GPUs
- **Cosine LR schedule with warmup** — linear ramp for 200 steps, cosine decay to 10% of peak
- **Gradient clipping** — norm clipped to 1.0 to prevent spikes

Checkpoints are saved automatically whenever validation loss improves.

---

## Generation

```python
from generation import generate_text

output = generate_text(
    model,
    tokenizer,
    prompt="DK",
    max_new_tokens=200,
    temperature=0.8,
    top_k=40,
)
print(output)
```

**Temperature guide:**

| Value | Character              |
|-------|------------------------|
| 0.5   | Focused, repetitive, safe |
| 0.8   | Balanced — good default |
| 1.0   | Full distribution, varied |
| 1.2+  | Creative, occasionally incoherent |

---

## Scaling Up

The default config is intentionally small for fast iteration. To scale:

**Bigger model:**
```python
cfg.d_model    = 768
cfg.n_layers   = 12
cfg.n_heads    = 12
# ~125M params — GPT-2 small scale
```

**Compile for faster training (PyTorch 2.0+):**
```python
model = torch.compile(model)
```

**FlashAttention (Ampere GPU or newer):**
```bash
pip install flash-attn
```
PyTorch's `scaled_dot_product_attention` dispatches to it automatically.

**Multi-GPU:**
```bash
torchrun --nproc_per_node=4 train.py
```
Wrap the model in `DistributedDataParallel` before training.

**Fine-tuning with LoRA:**
```bash
pip install peft
```
Freeze the base weights and train low-rank adapters on instruction data.

---

## Checkpoints

Save manually at any point:

```python
torch.save({
    "step":            step,
    "model_state":     model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "cfg":             cfg,
    "val_loss":        val_loss,
}, "dk_checkpoint.pt")
```

Load and resume:

```python
ckpt  = torch.load("dk_checkpoint.pt", map_location=device)
model = DKLanguageModel(ckpt["cfg"]).to(device)
model.load_state_dict(ckpt["model_state"])
```

---

## Requirements

| Package    | Version   |
|------------|-----------|
| Python     | 3.10+     |
| PyTorch    | 2.0+      |
| numpy      | any recent |
| matplotlib | any recent |
| tiktoken   | optional, for BPE tokenizer |
| flash-attn | optional, for FlashAttention kernel |

---

## License

MIT License. Free to use, modify, and distribute.

---

*Project DK — built from scratch by Puneeth Ram.*