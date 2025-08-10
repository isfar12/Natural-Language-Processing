# Question Answering with DistilBERT: Extractive QA System

This project implements an extractive question answering system using DistilBERT on the SQuAD dataset, handling complex challenges like long contexts, answer span extraction, and proper evaluation metrics.

## 📋 Project Overview

**Objective**: Build an extractive question answering system that can find answer spans within context paragraphs using DistilBERT, handling real-world challenges like context length limits and precise answer extraction.

**Key Components**:
- Extractive question answering with span prediction
- Context truncation with sliding windows
- Answer position alignment and extraction
- Comprehensive evaluation with SQuAD metrics

## 🛠️ Implementation Details

### 1. Dataset Loading and Analysis
```python
from datasets import load_dataset

data = load_dataset("squad")
```

**SQuAD Dataset Structure**:
```python
{
    'context': "The paragraph containing the answer",
    'question': "The question to be answered", 
    'answers': {
        'text': ["Answer span"],
        'answer_start': [character_position]
    }
}
```

**Key Insights**:
- **Training**: Usually 1 answer per question
- **Validation**: Multiple valid answers from different annotators
- **Evaluation**: Must compare against all possible correct answers

### 2. Understanding DistilBERT for QA
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

**BERT Input Format for QA**:
```
[CLS] question tokens [SEP] context tokens [SEP]
```

**Model Outputs**:
- **Start Logits**: Probability each token is the START of the answer
- **End Logits**: Probability each token is the END of the answer

### 3. Handling Long Contexts
```python
# Problem: Many contexts exceed 512 token limit
inputs = tokenizer(
    question,
    context,
    max_length=384,
    truncation="only_second",  # Only truncate context, keep full question
    stride=128,                # Overlap between chunks
    return_overflowing_tokens=True,
    return_offsets_mapping=True
)
```

**Sliding Window Strategy**:
- **max_length=384**: Each chunk has maximum 384 tokens
- **stride=128**: 128 tokens overlap between consecutive chunks  
- **return_overflowing_tokens=True**: Creates multiple inputs from long contexts
- **return_offsets_mapping=True**: Maps tokens back to character positions

### 4. Critical Concept: Offset Mapping
```python
# Offset mapping bridges tokens ↔ characters
offset_mapping = inputs["offset_mapping"]
# Example: [(0, 0), (0, 3), (4, 7), (8, 12), ...]
#          [CLS]   "The"   "cat"   "sat"
```

**Why It's Essential**:
- **SQuAD answers**: Given as character positions in original text
- **Model predictions**: Token positions in tokenized input
- **Offset mapping**: Converts between these two formats

### 5. Data Preprocessing Pipeline
```python
def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    
    start_positions = []
    end_positions = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        
        sequence_ids = inputs.sequence_ids(i)
        
        sample_index = sample_map[i]
        answers = examples["answers"][sample_index]
        
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            
            # Find token span that matches character span
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
                
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
                
            # Check if answer is in this chunk
            if not (offsets[token_start_index][0] <= start_char and 
                    offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                # Find exact token positions
                while (token_start_index < len(offsets) and 
                       offsets[token_start_index][0] <= start_char):
                    token_start_index += 1
                start_positions.append(token_start_index - 1)
                
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    
    return inputs
```

**Complex Logic Explained**:
1. **Sample Mapping**: Track which chunk belongs to which original example
2. **Sequence IDs**: Distinguish question tokens (0) from context tokens (1)
3. **Character-to-Token**: Convert answer character positions to token indices
4. **Boundary Checking**: Ensure answer span is within current chunk
5. **Impossible Answers**: Point to [CLS] token when answer not in chunk

### 6. Model Setup
```python
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
```

**Architecture**: DistilBERT + two classification heads (start/end position prediction).

### 7. Training Configuration
```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="distilbert-finetuned-squad",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)
```

### 8. Advanced Evaluation Pipeline
```python
def preprocess_validation_examples(examples):
    # Similar to training but keep example IDs for evaluation
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    
    for i in range(len(inputs["input_ids"])):
        sample_index = sample_map[i]
        example_ids.append(examples["id"][sample_index])
        
        sequence_ids = inputs.sequence_ids(i)
        context_index = 1
        inputs["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(inputs["offset_mapping"][i])
        ]
    
    inputs["example_id"] = example_ids
    return inputs
```

### 9. Answer Extraction from Predictions
```python
def postprocess_qa_predictions(examples, features, predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = predictions
    
    # Build mapping from example to features
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    
    predictions = collections.OrderedDict()
    
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        
        prelim_predictions = []
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            
            # Get best start and end positions
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (offset_mapping[start_index] is None or 
                        offset_mapping[end_index] is None):
                        continue
                    if (end_index < start_index or 
                        end_index - start_index + 1 > max_answer_length):
                        continue
                    
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    prelim_predictions.append({
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": example["context"][start_char:end_char]
                    })
        
        # Select best prediction
        predictions[example["id"]] = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[0]["text"]
    
    return predictions
```

**Post-processing Steps**:
1. **Score Calculation**: Combine start and end logits
2. **Span Validation**: Ensure valid start ≤ end relationships
3. **Length Filtering**: Remove overly long answers
4. **Character Extraction**: Use offset mapping to extract text
5. **Best Selection**: Choose highest-scoring valid span

### 10. SQuAD Evaluation Metrics
```python
def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    decoded_predictions = postprocess_qa_predictions(eval_examples, eval_dataset, predictions.predictions)
    
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]
    
    return metric.compute(predictions=formatted_predictions, references=references)
```

**SQuAD Metrics**:
- **Exact Match (EM)**: Percentage of predictions exactly matching any reference answer
- **F1 Score**: Token-level overlap between prediction and references

## 🔑 Key Technical Challenges

### 1. Context Length Management
**Challenge**: Real contexts often exceed BERT's 512 token limit.

**Solution**: Sliding window with stride creates overlapping chunks, ensuring answers aren't missed at boundaries.

### 2. Answer Position Alignment  
**Challenge**: SQuAD provides character positions, but models work with tokens.

**Solution**: Offset mapping provides bidirectional character ↔ token conversion.

### 3. Impossible Answer Detection
**Challenge**: Some chunks don't contain the answer.

**Solution**: Point start/end positions to [CLS] token for impossible answers.

### 4. Multiple Reference Evaluation
**Challenge**: Validation examples have multiple valid answers.

**Solution**: SQuAD metrics compare predictions against all references, taking the best match.

## 📊 Model Performance

The fine-tuned model achieves:
- **Exact Match**: ~80% (percentage of exact string matches)
- **F1 Score**: ~87% (token overlap with references)

**Performance Factors**:
- **Question Type**: Factual questions perform better than reasoning questions
- **Answer Length**: Shorter answers generally more accurate
- **Context Position**: Answers near beginning/end sometimes easier to find

## 🚀 Applications

### Real-World Use Cases:
- **Customer Support**: Automated FAQ systems
- **Document Search**: Find specific information in large documents  
- **Educational Tools**: Reading comprehension assessment
- **Legal Research**: Case law and statute analysis
- **Medical Information**: Clinical decision support

### Advanced Applications:
- **Multi-hop QA**: Questions requiring multiple reasoning steps
- **Conversational QA**: Context-aware dialogue systems
- **Visual QA**: Combined text and image understanding
- **Cross-lingual QA**: Questions and answers in different languages

## 💡 Key Learnings

1. **Extractive vs Generative**: Extractive QA finds existing spans, generative creates new text
2. **Context Handling**: Sliding windows essential for long documents
3. **Alignment Complexity**: Character-token mapping requires careful handling
4. **Evaluation Nuances**: Multiple valid answers need sophisticated metrics
5. **Production Considerations**: Speed vs accuracy tradeoffs for deployment

### Architecture Insights:
- **Dual Output Heads**: Separate prediction of start and end positions
- **Position Embeddings**: Help model understand token relationships
- **Attention Patterns**: Model learns to focus on relevant context regions
- **Transfer Learning**: Pre-trained language understanding crucial for performance

This project demonstrates building production-ready question answering systems that can handle real-world complexity while maintaining high accuracy and efficiency.
