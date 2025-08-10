# Text Generation with GPT-2: Poetry Generation

This project demonstrates fine-tuning GPT-2 for causal language modeling to generate poetry-style text, including data preprocessing, model training, and text generation pipelines.

## 📋 Project Overview

**Objective**: Fine-tune GPT-2 on poetry text to create a specialized text generation model capable of generating creative, poetry-style content.

**Key Components**:
- Causal language modeling with GPT-2
- Text preprocessing and chunking
- Block-wise data organization
- Creative text generation pipeline

## 🛠️ Implementation Details

### 1. Data Loading and Preprocessing
```python
from pathlib import Path
import pandas as pd

# Load poetry text data
poem_path = Path("../../Dataset/poems.txt")
poem_text = poem_path.read_text(encoding="utf-8")

# Create DataFrame for processing
df = pd.DataFrame({'text': [poem_text]})
```

**Data Source**: Large poetry text corpus for training the language model.

### 2. Tokenization Setup
```python
from transformers import AutoTokenizer

# Load GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

def tokenize_function(example):
    return tokenizer(example["text"])

# Convert to Hugging Face Dataset and tokenize
from datasets import Dataset
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
```

**Key Setup**:
- GPT-2 doesn't have a padding token, so we use the EOS token
- Batch tokenization for efficiency
- Remove original text column after tokenization

### 3. Text Chunking Strategy
```python
# Concatenate all tokens and chunk into fixed-size blocks
concatenated = sum(tokenized_dataset['input_ids'], [])
print(f"Total tokens: {len(concatenated)}")

block_size = 128  # Sequence length for training

def group_texts(examples):
    # Concatenate all input_ids together
    concatenated = sum(examples["input_ids"], [])
    total_length = (len(concatenated) // block_size) * block_size
    
    result = {
        "input_ids": [concatenated[i : i + block_size] for i in range(0, total_length, block_size)],
    }
    result["attention_mask"] = [[1] * block_size] * len(result["input_ids"])
    result["labels"] = result["input_ids"].copy()  # For language modeling, labels = inputs
    
    return result

# Apply chunking
lm_datasets = tokenized_dataset.map(group_texts, batched=True)
```

**Chunking Logic**:
- **Concatenation**: Join all text into one long sequence
- **Block Division**: Split into fixed-size chunks (128 tokens)
- **Label Creation**: For causal LM, labels are the same as inputs (shifted internally)
- **Attention Masks**: All tokens are real (no padding within chunks)

### 4. Model Loading
```python
from transformers import GPT2LMHeadModel

# Load pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
```

**Architecture**: GPT-2 with causal language modeling head for next-token prediction.

### 5. Training Configuration
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="poem_gpt2",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    prediction_loss_only=True
)
```

**Configuration Notes**:
- Small batch size (2) to handle memory constraints
- Regular checkpointing every 100 steps
- Only track prediction loss for simplicity

### 6. Data Collation
```python
from transformers import DataCollatorForLanguageModeling

# Collator for causal language modeling (not masked LM)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  # GPT-2 is causal, not masked LM
)
```

**Key Difference**: `mlm=False` because GPT-2 uses causal (next-token) prediction, not masked language modeling like BERT.

### 7. Training Setup
```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()
```

### 8. Model Persistence
```python
# Save the fine-tuned model
trainer.save_model("poem")
tokenizer.save_pretrained("poem")
```

### 9. Text Generation Pipeline
```python
from transformers import pipeline

# Create generation pipeline
generator = pipeline("text-generation", model="poem", tokenizer=tokenizer)

# Generate poetry
lines = generator("love", max_length=100, num_return_sequences=1)
print(lines[0]['generated_text'])
```

**Generation Parameters**:
- **Prompt**: Starting word/phrase ("love")
- **max_length**: Maximum total tokens (including prompt)
- **num_return_sequences**: Number of different completions

## 🔑 Key Technical Concepts

### Causal Language Modeling
**Objective**: Predict the next token given previous tokens.

**Training Process**:
1. **Input**: `[token1, token2, token3, token4]`
2. **Targets**: `[token2, token3, token4, token5]` (shifted by 1)
3. **Prediction**: Model learns P(token_i | token_1, ..., token_{i-1})

### Block-wise Training Strategy
**Why 128-token blocks?**
- **Memory Efficiency**: Fixed-size batches
- **Context Length**: Reasonable context for poetry
- **Training Stability**: Consistent sequence lengths

**Data Flow**:
```
Original: "roses are red violets are blue..."
Tokenized: [1234, 389, 2134, 8923, 389, 2156, ...]
Chunked: [[1234, 389, 2134, ...], [8923, 389, 2156, ...]]
```

### GPT-2 Architecture Benefits
- **Autoregressive**: Natural for text generation
- **Transformer Decoder**: Strong pattern learning
- **Pre-trained**: Rich language understanding
- **Scalable**: Can handle longer contexts

## 📊 Generation Examples

The fine-tuned model can generate poetry-style text:

**Example Generation**:
```
Input: "love"
Output: "love is like a summer breeze
         that whispers through the trees
         and dances in the morning light..."
```

**Quality Factors**:
- **Coherence**: Maintains thematic consistency
- **Style**: Adopts poetry-like structure
- **Creativity**: Generates novel combinations
- **Relevance**: Stays on topic

## 🚀 Applications

### Creative Writing:
- **Poetry Generation**: Automated poetry creation
- **Songwriting**: Lyric generation
- **Story Prompts**: Creative writing assistance
- **Content Creation**: Blog posts, social media

### Technical Applications:
- **Data Augmentation**: Generate training data
- **Style Transfer**: Adapt text to specific styles
- **Interactive Fiction**: Dynamic story generation
- **Educational Tools**: Creative writing assistance

## 🔧 Advanced Features

### Generation Control:
- **Temperature**: Control randomness (0.0-2.0)
- **Top-p Sampling**: Nucleus sampling for quality
- **Top-k Sampling**: Limit vocabulary selection
- **Repetition Penalty**: Reduce repetitive text

```python
# Advanced generation
advanced_output = generator(
    "moonlight",
    max_length=100,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True
)
```

### Fine-tuning Variations:
- **Domain-specific**: Different poetry styles (haiku, sonnets)
- **Author-specific**: Mimic particular poets
- **Multi-modal**: Combine with image inputs
- **Interactive**: Real-time generation systems

## 💡 Key Learnings

1. **Causal Language Modeling**: Next-token prediction for text generation
2. **Data Preprocessing**: Importance of proper text chunking
3. **GPT-2 Architecture**: Autoregressive transformer design
4. **Generation Control**: Parameters for quality and creativity
5. **Creative AI**: Balancing coherence and novelty

### Challenges Addressed:
- **Memory Management**: Efficient batching strategies
- **Data Organization**: Proper sequence preparation
- **Model Adaptation**: Fine-tuning for specific domains
- **Quality Control**: Generation parameter tuning

This project demonstrates the complete pipeline for building creative text generation systems, from data preparation to deployment-ready models for artistic and practical applications.
