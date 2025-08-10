# Machine Translation: English to Spanish

This project demonstrates fine-tuning a pre-trained Helsinki-NLP translation model for English-Spanish translation using sequence-to-sequence learning with the Hugging Face Transformers library.

## 📋 Project Overview

**Objective**: Fine-tune the Helsinki-NLP/opus-mt-en-es model on the KDE4 dataset to improve English-Spanish translation performance.

**Key Components**:
- Sequence-to-sequence model training
- Translation-specific tokenization
- Data collation for seq2seq models
- BLEU and BERTScore evaluation
- Translation pipeline creation

## 🛠️ Implementation Details

### 1. Dataset Loading
```python
from datasets import load_dataset

# Load KDE4 English-Spanish translation dataset
data = load_dataset("kde4", lang1="en", lang2="es", trust_remote_code=True)
data = data["train"].shuffle(seed=42).select(range(1000))  # Subset for training
data = data.train_test_split(test_size=0.1)
```

**Dataset**: KDE4 contains English-Spanish translation pairs from KDE software documentation.

### 2. Specialized Tokenization
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

def tokenization_function(data):
    inputs = [x["en"] for x in data["translation"]]
    targets = [x["es"] for x in data["translation"]]

    tokenized_inputs = tokenizer(inputs, truncation=True, max_length=max_input_length)
    tokenized_outputs = tokenizer(text_target=targets, truncation=True, max_length=max_output_length)

    tokenized_inputs["labels"] = tokenized_outputs["input_ids"]
    return tokenized_inputs
```

**Key Features**:
- Language-pair specific tokenizer
- Separate handling of source and target languages
- Dynamic length constraints based on data analysis

### 3. Data Length Analysis
```python
import matplotlib.pyplot as plt

# Analyze input lengths to set appropriate max_length
train = data["train"]["translation"]
input_lens = [len(tokenizer(data["en"])["input_ids"]) for data in train]
plt.hist(input_lens, bins=50)
plt.xlabel("Input Length")
```

**Optimization**: Histogram analysis shows most sequences are under 128 tokens, allowing efficient max_length selection.

### 4. Seq2Seq Data Collation
```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

**Critical Functions**:
- **Dynamic Padding**: Pads sequences to batch maximum length
- **Decoder Input Creation**: Automatically generates `decoder_input_ids` by shifting labels
- **Teacher Forcing**: Enables model to see correct previous tokens during training

### 5. Model Loading
```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
```

### 6. Advanced Evaluation Metrics
```python
from evaluate import load
sacrebleu = load("sacrebleu")
bertscore = load("bertscore")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    sacrebleu_result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    bertscore_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="es")
    
    return {
        "bleu": sacrebleu_result["score"],
        "bertscore_precision": np.mean(bertscore_result["precision"]),
        "bertscore_recall": np.mean(bertscore_result["recall"]),
    }
```

**Metrics Used**:
- **BLEU Score**: N-gram overlap between prediction and reference
- **BERTScore**: Semantic similarity using contextual embeddings

### 7. Training Configuration
```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="opus_mt_en_es",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
```

### 8. Training Process
```python
# Pre-training evaluation
trainer.evaluate(max_length=max_output_length)

# Fine-tuning
trainer.train()

# Post-training evaluation
trainer.evaluate(max_length=max_output_length)

# Save the model
trainer.save_model("opus_mt_en_es_final")
```

### 9. Translation Pipeline
```python
from transformers import pipeline

translator = pipeline("translation", model=model, tokenizer=tokenizer)
result = translator("This is a test sentence to translate.", max_length=max_output_length)
print(result)
```

## 🔑 Key Concepts

### Teacher Forcing in Seq2Seq
**Understanding the Data Collator Magic**:

```python
# Example of teacher forcing alignment
decoder_input_ids: [</s>, "Como", "se", "llama", ...]  # What decoder sees
labels:           ["Como", "se", "llama", "?", </s>]    # What it should predict
```

This offset enables the model to learn next-token prediction with ground truth context.

### Why Helsinki-NLP Tokenizer?
1. **Language-Pair Specific**: Trained on English-Spanish translation pairs
2. **Subword Tokenization**: Handles out-of-vocabulary words efficiently
3. **Model Compatibility**: Vocabulary matches pre-trained model exactly
4. **Special Tokens**: Includes necessary seq2seq markers

## 📊 Evaluation Results

The model evaluation provides:
- **BLEU Scores**: Industry standard translation quality metric
- **BERTScore**: Modern semantic similarity measurement
- **Before/After Comparison**: Shows improvement from fine-tuning

## 🚀 Applications

### Real-World Use Cases:
- **Document Translation**: Technical documentation translation
- **Website Localization**: Multi-language website content
- **Customer Support**: Real-time translation for global support
- **Content Creation**: Bilingual content generation

### Extension Possibilities:
- **Multi-language Support**: Extend to other language pairs
- **Domain Adaptation**: Fine-tune on specific domains (medical, legal)
- **Interactive Translation**: Build web applications with the pipeline
- **Quality Estimation**: Add confidence scoring

## 💡 Key Learnings

1. **Seq2Seq Architecture**: Understanding encoder-decoder models
2. **Translation Metrics**: BLEU vs. semantic evaluation approaches
3. **Data Collation**: Critical for proper seq2seq training
4. **Language-Specific Considerations**: Tokenizer selection importance
5. **Evaluation Strategy**: Comprehensive metric selection for translation quality

This project demonstrates the complete pipeline for neural machine translation, from data preprocessing to deployment-ready models.
