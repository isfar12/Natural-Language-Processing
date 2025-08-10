# Textual Entailment Classification with BERT

This project implements textual entailment classification using BERT to determine whether a hypothesis sentence logically follows from a premise sentence, using the GLUE RTE (Recognizing Textual Entailment) dataset.

## 📋 Project Overview

**Objective**: Build a binary classifier to determine if sentence pairs have an entailment relationship (Yes/No) using BERT for sequence classification.

**Key Components**:
- Multiple input sentence handling
- BERT-based sequence pair classification  
- Custom configuration for label mapping
- Inference pipeline for new predictions

## 🛠️ Implementation Details

### 1. Dataset Loading
```python
from datasets import load_dataset

data = load_dataset("glue", "rte")
```

**Dataset**: GLUE RTE (Recognizing Textual Entailment) contains premise-hypothesis pairs with binary entailment labels.

**Data Structure**:
```python
{
  'sentence1': "The premise sentence",
  'sentence2': "The hypothesis sentence", 
  'label': 0 or 1  # 0: No entailment, 1: Entailment
}
```

### 2. Tokenization for Sentence Pairs
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(data):
    return tokenizer(data["sentence1"], data["sentence2"], truncation=True)

tokenized_data = data.map(tokenize_function, batched=True)
```

**Key Features**:
- **Dual Input Processing**: Handles sentence pairs automatically
- **BERT Format**: Creates `[CLS] sentence1 [SEP] sentence2 [SEP]` structure
- **Automatic Truncation**: Manages long sentence pairs

### 3. Label Configuration
```python
from transformers import AutoConfig

label2id = {"No": 0, "Yes": 1} 
id2label = {v: k for k, v in label2id.items()}

config = AutoConfig.from_pretrained(
    "bert-base-uncased",
    label2id=label2id,
    id2label=id2label
)
```

**Configuration Purpose**: Maps numeric labels to human-readable categories for better interpretability.

### 4. Model Setup
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    config=config
)
```

**Architecture**: BERT encoder + classification head for binary prediction.

### 5. Training Configuration
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="result",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    logging_steps=150
)
```

### 6. Evaluation Metrics
```python
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    pred = predictions.argmax(axis=1)
    accuracy = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred)
    return {
        "accuracy": accuracy,
        "f1": f1
    }
```

**Metrics Used**:
- **Accuracy**: Overall classification correctness
- **F1-Score**: Balanced measure for binary classification

### 7. Training Process
```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()
```

### 8. Model Persistence
```python
model.save_pretrained("model_directory")
tokenizer.save_pretrained("model_directory")
```

### 9. Inference Pipeline
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

def load_trained_model(model_path="model_directory"):
    """Load the trained model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

def predict_entailment(sentence1, sentence2, model, tokenizer):
    """Predict whether sentence1 entails sentence2"""
    inputs = tokenizer(sentence1, sentence2, 
                      return_tensors="pt", 
                      truncation=True, 
                      padding=True, 
                      max_length=512)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
    
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()
    
    id2label = {0: "No", 1: "Yes"}
    predicted_label = id2label[predicted_class]
    
    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "probabilities": {
            "No": probabilities[0][0].item(),
            "Yes": probabilities[0][1].item()
        }
    }

# Usage example
trained_model, trained_tokenizer = load_trained_model("model_directory")
result = predict_entailment(
    "I saw a movie.", 
    "The movie was good", 
    trained_model, 
    trained_tokenizer
)
```

## 🎯 Understanding Textual Entailment

### What is Textual Entailment?
**Entailment** occurs when the truth of sentence A guarantees the truth of sentence B.

**Examples**:
- ✅ **Entailment**: "John is running" → "John is moving" 
- ❌ **No Entailment**: "John is running" → "John is happy"

### Classification Categories:
- **Yes (1)**: Hypothesis logically follows from premise
- **No (0)**: Hypothesis does not follow from premise

## 🔑 Key Technical Concepts

### BERT for Sentence Pairs
**Input Format**: `[CLS] sentence1 [SEP] sentence2 [SEP]`

**How BERT Processes Pairs**:
1. **Token Embeddings**: Convert words to vectors
2. **Segment Embeddings**: Distinguish between sentences
3. **Position Embeddings**: Encode token positions
4. **Self-Attention**: Model relationships between all tokens
5. **Classification Head**: Binary decision from `[CLS]` token

### Model Architecture Benefits:
- **Bidirectional Context**: Understands both sentences simultaneously  
- **Cross-Sentence Attention**: Models interactions between sentences
- **Pre-trained Knowledge**: Leverages BERT's language understanding

## 📊 Applications

### Real-World Use Cases:
- **Information Retrieval**: Find relevant documents
- **Question Answering**: Verify answer consistency
- **Fact Checking**: Detect contradictory information
- **Reading Comprehension**: Assess text understanding
- **Legal Document Analysis**: Analyze contract implications

### Advanced Applications:
- **Multi-hop Reasoning**: Chain multiple entailment steps
- **Contradiction Detection**: Identify conflicting statements
- **Semantic Search**: Find semantically related content
- **Dialogue Systems**: Ensure response consistency

## 🚀 Extension Possibilities

### Model Improvements:
- **RoBERTa/DeBERTa**: More powerful base models
- **Domain-Specific Fine-tuning**: Legal, medical, scientific texts
- **Multi-class Classification**: Entailment/Neutral/Contradiction
- **Ensemble Methods**: Combine multiple models

### Data Augmentation:
- **Paraphrase Generation**: Create more training examples
- **Adversarial Examples**: Improve robustness
- **Cross-lingual Transfer**: Multiple languages

## 💡 Key Learnings

1. **Sentence Pair Processing**: How BERT handles multiple inputs
2. **Configuration Management**: Custom label mapping
3. **Binary Classification**: Evaluation metrics selection
4. **Model Inference**: Production-ready prediction pipeline
5. **Practical NLU**: Real-world natural language understanding task

This project demonstrates building robust natural language understanding systems for determining logical relationships between text pairs, a fundamental capability for many AI applications.
