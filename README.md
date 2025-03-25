# Healthcare Summarization Project

## Project Overview
This project focuses on generating multi-perspective summaries from medical-related user questions and answers using state-of-the-art language models.

## Configuration

### Configuration File (`config.yaml`)
The configuration file contains key settings for the project:

```yaml
# Healthcare Summarization Configuration

# Data Settings
data_path: "./data"
output_path: "./results"
datasets:
 - "test.json"
 - "valid.json"
max_samples: null # Set to a number for testing or null to process all samples

# Model Settings
model_size: "8B" # Options: "8B" or "70B"
device: "cuda" # Options: "cuda" or "cpu"
quantization: null # Options: null, "8bit", "4bit"
model_name: "meta-llama/Llama-3.2-3B-Instruct"

# Generation Settings
max_new_tokens: 50000
temperature: 0.1
do_sample: true

# System Settings
batch_processing: false # Future feature for batch processing
cache_dir: null # Optional custom cache directory for models
```

### Running the Project
To run the main script:
```bash
python main.py --config ./config.yaml
```

## Project Structure
```
project_root/
│
├── data/
│   ├── train.json
│   ├── test.json
│   └── valid.json
│
├── results/
│   ├── run_config.yaml
│   ├── summary_report.csv
│   ├── test_metrics.json
│   ├── test_results.json
│   ├── valid_metrics.json
│   └── valid_results.json
│
├── main.py
├── config.yaml
└── requirements.txt
```

## Data Format

### Input Data (JSON)
Each JSON file contains a list of dictionaries with the following structure:
```json
[
  {
    "uri": "unique_identifier",
    "question": "Medical question",
    "context": "Additional context",
    "answers": ["answer1", "answer2", ...],
    "labelled_answer_spans": {
      "EXPERIENCE": [...],
      "CAUSE": [...],
      "INFORMATION": [...]
    }
  }
]
```

## Results Format

### Metrics Files (`*_metrics.json`)
```json
{
  "dataset": "test.json",
  "model": "meta-llama/Llama-3.2-3B-Instruct",
  "samples": 640,
  "INFORMATION_BLEU_AVG": 0.0031927,
  "INFORMATION_BERT_AVG": 0.8827259,
  "CAUSE_BLEU_AVG": 0.0030258,
  "CAUSE_BERT_AVG": 0.8883831,
  "SUGGESTION_BLEU_AVG": 0.0036648,
  "SUGGESTION_BERT_AVG": 0.8797409,
  "EXPERIENCE_BLEU_AVG": 0.0032600,
  "EXPERIENCE_BERT_AVG": 0.8584677,
  "OVERALL_BLEU_AVG": 0.0034157,
  "OVERALL_BERT_AVG": 0.8781039
}
```

### Results Files (`*_results.json`)
```json
[
  {
    "uri": "unique_identifier",
    "question": "Medical question",
    "context": "Additional context",
    "INFORMATION_REFERENCE": "Reference summary",
    "INFORMATION_GENERATED": "Model-generated summary",
    "INFORMATION_BLEU": 0.002660,
    "INFORMATION_BERT": 0.882331,
    "AVG_BLEU": 0.002660,
    "AVG_BERT": 0.882331
  }
]
```

## Key Components

### Input Formatting
The project uses a custom input formatting function:
```python
def format_input(question, context, answers, perspective, excerpt):
    # Formats input for the language model
    # Includes question, context, perspective, and answer excerpts
```

### Function Definitions
Four key summary generation functions are defined:
1. `generate_information_summary`
2. `generate_cause_summary`
3. `generate_suggestion_summary`
4. `generate_experience_summary`

### System Prompt
A detailed system prompt guides the language model in generating summaries across different perspectives.

## Evaluation Metrics
- BLEU Score: Measures the precision of generated summaries
- BERT Score: Assesses semantic similarity between generated and reference summaries

## Model
- Model: Llama 3.2 3B Instruct
- Evaluated on validation and test datasets

## Requirements
- Python 3.12+
- PyTorch 2.6.0+
- Transformers 4.50.0+
- Bert Score 0.3.13+
- Numpy 2.2.4+
- Pandas 2.2.3+
- NLTK 3.9.1+

## Installation
```bash
pip install -r requirements.txt
```

## Contributors

### Core Development Team
- **Shashank Sharma** 
  - GitHub: [@shashank23088](https://github.com/shashank23088)
  - Email: shashank23088@iiitd.ac.in
  - Roll No: MT23088

- **Shreyas Gupta**
  - GitHub: [@Shreyas-Gupta-IIITD](https://github.com/Shreyas-Gupta-IIITD)
  - Email: shreyas23221@iiitd.ac.in
  - Roll No: MT23221
 
- **Sajid Javid**
  - GitHub: [@sajidjavid222](https://github.com/sajidjavid222)
  - Email: sajidj@iiitd.ac.in
<<<<<<< HEAD
  - Roll No: PhD24002
=======
  - Roll No: PhD24002
>>>>>>> bb328a3 (updated contributions)
