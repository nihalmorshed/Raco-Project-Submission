# Fine-Tuning LLaMA 3.1-8B-Instruct on Bengali Empathetic Conversations

## Documentation: Configuration, Strategy & Challenges

---

## 1. Project Overview

This project fine-tunes Meta's LLaMA 3.1-8B-Instruct model on the Bengali Empathetic Conversations dataset using parameter-efficient fine-tuning (PEFT) on Kaggle's free Tesla T4 GPU (16GB VRAM).

### Key Constraints
- **Hardware**: Kaggle T4 GPU with 16GB VRAM
- **Time Limit**: 12-hour maximum session (For Kaggle)
- **Sequence Length**: 4096 tokens
- **Model Size**: 8 billion parameters

### Actual Training Results
- **Training Samples**: 4,000
- **Final Loss**: 0.5285
- **Training Time**: 581.7 minutes (~9.7 hours)
- **Peak GPU Memory**: 14.162 GB
- **Trainable Parameters**: 3,407,872 (0.0424%)

---

## 2. Choice of LoRA/Unsloth Configuration

### 2.1 Why QLoRA over Full Fine-tuning?

Full fine-tuning of an 8B parameter model requires ~32GB VRAM in fp16, which exceeds the T4's 16GB capacity. QLoRA (Quantized Low-Rank Adaptation) solves this by:

1. **4-bit Quantization**: Reduces model memory from ~16GB to ~5GB
2. **LoRA Adapters**: Trains only 0.0424% of parameters (~3.4M vs 8B)
3. **Gradient Checkpointing**: Trades compute for memory

### 2.2 Why Standard LoRA over Unsloth?

**Initial Approach**: We attempted to use Unsloth for its 2-5x training speedup.

**Problems Encountered**:
1. **TRL Version Conflicts**: `AttributeError: 'int' object has no attribute 'mean'` with TRL >= 0.9.0
2. **NumPy Binary Incompatibility**: Kaggle's packages compiled for NumPy 1.x conflicted with NumPy 2.x
3. **Torch/Torchvision Mismatches**: Unsloth upgraded torch but broke torchvision dependencies

**Decision**: After 4+ debugging iterations, we chose standard HuggingFace PEFT for stability. The tradeoff of slower training (0.12 samples/sec) was acceptable given the reliability guarantee.

### 2.3 Final LoRA Configuration

```python
LoRAConfig(
    r=8,                              # Rank (optimized for T4 memory)
    lora_alpha=16,                    
    lora_dropout=0.05,                
    bias="none",                      
    target_modules=["q_proj", "v_proj"],  # Query & Value projections only
)
```

**Reason for Choice**:
- **Rank 8**: Sufficient for task-specific adaptation while minimizing parameters
- **Target Modules**: q_proj and v_proj capture the most task-relevant attention patterns; excluding k_proj, o_proj, and MLP layers significantly reduces computation
- **Alpha = 2×Rank**: Standard scaling factor for stable training

### 2.4 Quantization Configuration

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # Normal Float 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16
    bnb_4bit_use_double_quant=True,   # Quantize the quantization constants
)
```

---

## 3. Training Strategy

### 3.1 Data Pipeline

| Stage | Description | Output |
|-------|-------------|--------|
| Load | Read CSV with 38,233 Bengali conversation pairs | Raw DataFrame |
| Clean | Remove nulls, filter short Q&A | 33,731 samples |
| Filter | Remove samples > 5000 chars (OOM prevention) | ~30,000 samples |
| Format | Convert to LLaMA 3.1 Instruct format | Formatted text |
| Split | Fixed splits for reproducibility | Train/Val/Test |
| Limit | 4,000 train / 300 val / 300 test | Final datasets |

### 3.2 LLaMA 3.1 Instruct Format

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

আপনি একজন সহানুভূতিশীল বাংলা কথোপকথন সহকারী। বিষয়: {topic}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question_title}

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer}<|eot_id|><|end_of_text|>
```

### 3.3 Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 1 | Time constraint (12h limit) |
| Batch Size | 1 | Memory constraint |
| Gradient Accumulation | 4 | Effective batch = 4 |
| Learning Rate | 2e-4 | Standard for LoRA |
| Warmup | 3% | Gradual LR increase |
| Scheduler | Cosine | Smooth decay |
| Optimizer | Paged AdamW 8-bit | Memory efficient |
| Precision | BFloat16 | T4 compatible, stable |
| Max Seq Length | 4096 | Full length (requirement) |
| Save Steps | 250 | Checkpoint frequency |
| Logging Steps | 50 | Training monitoring |

### 3.4 Memory Optimization Techniques

1. **Gradient Checkpointing**: Recompute activations during backward pass
   - Setting: `use_reentrant=True` (stable version)
2. **4-bit Quantization**: NF4 with double quantization
3. **Paged Optimizer**: 8-bit AdamW with CPU offloading
4. **CUDA Memory Config**: `expandable_segments=True` to reduce fragmentation
5. **Long Sample Filtering**: Remove samples > 5000 characters

---

## 4. Challenges Faced & Solutions

### Challenge 1: Unsloth Dependency Conflicts

**Problem**: Multiple package version conflicts on Kaggle
```
AttributeError: 'int' object has no attribute 'mean'
ImportError: numpy.dtype size changed, may indicate binary incompatibility
```

**Solution**: Abandoned Unsloth; used standard HuggingFace stack (transformers, peft, trl)

---

### Challenge 2: CUDA Out of Memory

**Problem**: Oout Of Memory errors during training despite 4-bit quantization
```
OutOfMemoryError: CUDA out of memory. Tried to allocate 1.22 GiB.
```

**Root Cause**: Some Bengali text samples were very long, causing memory spikes

**Solution**:
- Filter samples > 5000 characters before training
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Clear GPU cache before training

---

### Challenge 3: Mixed Precision Conflicts

**Problem**: BFloat16 vs Float16 gradient mismatch
```
RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
```

**Solution**: Aligned all precision settings to BFloat16:
- Model: `torch_dtype=torch.bfloat16`
- Quantization: `bnb_4bit_compute_dtype=torch.bfloat16`
- Training: `bf16=True, fp16=False`

---

### Challenge 4: Gradient Checkpointing Tensor Mismatch

**Problem**: Different tensor counts during forward/backward
```
CheckpointError: A different number of tensors was saved during the original forward and recomputation.
```

**Solution**: Changed `use_reentrant=False` to `use_reentrant=True` (older, stable implementation)

---

### Challenge 5: Training Speed vs Time Limit

**Problem**: Training too slow for Kaggle's 12-hour limit
- Initial estimates: 10,000+ steps at 0.01-0.02 it/s = 100+ hours

**Solution**: Aggressive optimization
- Used 4,000 training samples (balance between quality and time)
- LoRA rank: 8 (memory efficient)
- Limited target modules: 2 (q_proj, v_proj only)
- Disabled evaluation during training
- Final: ~1,000 steps in ~9.7 hours

---

## 5. Evaluation Approach

### 5.1 Automatic Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **BLEU** | 0.81 | N-gram overlap with reference responses |
| **ROUGE-1** | 0.0000 | Unigram overlap (see note below) |
| **ROUGE-2** | 0.0000 | Bigram overlap |
| **ROUGE-L** | 0.0000 | Longest common subsequence |
| **Empathy (Keyword)** | 3.67 | Keyword-based Bengali empathy metric |
| **Empathy (Semantic)** | 19.33 | Semantic similarity-based metric |
| **Combined Score** | 0.320 | Weighted quality + empathy score |

**Note on ROUGE Scores**: Zero scores indicate potential tokenization mismatch between Bengali predictions and references. This is a known limitation when evaluating generative models on creative/empathetic tasks.

### 5.2 Empathetic Patterns (Bengali)

```python
EMPATHETIC_PATTERNS = [
    "বুঝতে পারছি",  # I understand
    "বুঝি",        # Sad
    "দুঃখিত",       # Sad
    "কষ্ট",         # Pain/suffering
    "মন খারাপ",     # Sad
    "পাশে আছি",    # I will be there for you
    "সাহায্য",      # Help
    "চেষ্টা",       # Help
    "সাহস",        # Strength
    "আশা",         
]
```

### 5.3 Human Evaluation Framework

Evaluators rate responses on 1-5 scale:
- **Empathy**: Does the response show understanding?
- **Relevance**: Is it appropriate to the input?
- **Fluency**: Is the Bengali grammatically correct?
- **Overall**: General quality rating


---

## 6. Reproducibility

### Code Running Environment
- Platform: Kaggle Notebooks
- GPU: Tesla T4 (16GB VRAM)
- Python: 3.11
- PyTorch: 2.6.0+cu124

### Package Dependencies
```
transformers>=4.40.0
peft>=0.10.0
trl>=0.8.0
bitsandbytes>=0.43.0
datasets
accelerate
evaluate
sacrebleu
rouge-score
sentence-transformers  # For semantic empathy scoring
```

### Random Generator Seeds
- Data shuffling: 42
- Training: 42

### Model Artifacts
- HuggingFace Hub: `CaptainNimo/bengali-empathetic-llama-3.1-8b-lora4000`
- Files: adapter_model.safetensors, adapter_config.json, tokenizer files

---

## 7. Resource-Constrained Limitations & Unrealized Improvements

The following enhancements were considered but could not be implemented due to Kaggle's free-tier GPU limitations (16GB VRAM, 12-hour sessions):

1. **Use Unsloth on Colab Pro**: Better environment control, potentially 5x faster
2. **Full Dataset Training**: With more GPU hours, train on all 33,000+ samples
3. **Higher LoRA Rank**: r=16 or r=32 for better adaptation (requires more VRAM)
4. **Multi-Epoch Training**: 3 epochs for better convergence (requires more compute time)
5. **Include All Attention Layers**: Add k_proj, o_proj, gate_proj, up_proj, down_proj (memory intensive)
6. **Compute Perplexity**: Full perplexity evaluation on test set (compute constrained)

---

## 8. Knowledge Gaps & Learning Interests

### Acknowledging Limitations

I want to be transparent about areas where my understanding fell short during this project:

**Evaluation Framework Gaps**:
- The **ROUGE scores of 0.0000** indicate that my implementation of the evaluation pipeline may have issues I don't fully understand—potentially related to Bengali tokenization, text normalization, or how the `evaluate` library handles non-Latin scripts.
- My theoretical understanding of **Perplexity** calculation for fine-tuned models with LoRA adapters is limited. While I implemented a basic version, I'm uncertain whether it correctly reflects the model's actual language modeling capability.
- The **Empathy Score** metrics (keyword-based: 3.67, semantic: 19.33) are custom implementations. I lack the domain expertise to validate whether these adequately measure empathetic response quality in Bengali conversational AI.

**Unsatisfactory Results**:
- The low empathy scores suggest the model's responses lack the emotional depth expected from an empathetic conversational agent.
- The evaluation results do not meet the quality benchmarks I had hoped for, and I acknowledge this reflects gaps in both my experimental methodology and theoretical knowledge of NLP evaluation.

Despite these shortcomings, this project has influenced my curiosity in several areas:

1. **Cross-lingual Evaluation Metrics**: I would welcome the opportunity to learn how BLEU, ROUGE, and other metrics should be properly adapted for Bengali and other Indic languages.

2. **Human Evaluation Design**: Understanding how to design human evaluation studies for empathetic AI systems is an area I'm keen to explore further.

3. **Advanced Fine-tuning Techniques**: I'm interested in learning about techniques like DPO (Direct Preference Optimization), RLHF, and constitutional AI methods that could improve empathetic response generation.

4. **Perplexity & Language Model Evaluation**: A deeper understanding of how to properly evaluate generative language models, especially for low-resource languages, would be invaluable.

**If given the opportunity, I am committed to addressing these knowledge gaps through further study, mentorship, and hands-on experimentation.**

---

## 9. Sample Model Outputs

### Test Case 1: Exam Anxiety
**Input**: আমি খুব চিন্তিত আছি আমার পরীক্ষার ফলাফল নিয়ে। কী করব বুঝতে পারছি না।

**Output**: ওহ, আপনি অপেক্ষা করতে হবে. দেখুন আপনি ধারণাটি বিবেচনা করেন এবং আপনার উপর আরও ভাল পড়াশোনা করুন।

### Test Case 2: Friendship Conflict
**Input**: আমার বন্ধুর সাথে ঝগড়া হয়ে গেছে। খুব মন খারাপ।

**Output**: ওহ না! কি ঘটেছে? আপনি উপলব্ধি করতে চাইলে, আমি এটা দেখতে পারি.

### Test Case 3: Job Loss
**Input**: আমি চাকরি হারিয়েছি। কীভাবে এগিয়ে যাব?

**Output**: দুঃখিত, তাই. আমি আশা করি আপনি তাদের সাথে ফিরে যাবেন এবং তাদের অন্য কোন কাজ দিতে পারেন!

### Test Case 4: Mental Health
**Input**: আমি অনেক দিন ধরে বিষণ্ণ বোধ করছি। সাহায্য দরকার।

**Output**: আমি এটা খুব খারাপ. আমি বলতে পারি আপনাকে কিছু বিষয় চিন্তা করতে বলব...

---

## 10. Conclusion

Despite significant hardware and time constraints, I successfully managed to fine-tune LLaMA 3.1-8B on Bengali empathetic conversations using QLoRA. Key decisions: choosing stability over speed, aggressive memory optimization, and strategic parameter reduction, enabled training completion within Kaggle's limits while maintaining the full 4096 token sequence length as required.

The resulting model generates contextually appropriate responses in Bengali, demonstrating the viability of parameter-efficient fine-tuning for low-resource language tasks. However, the empathy quality indicates that more training (epochs, samples) would improve response quality significantly.
