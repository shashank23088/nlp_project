import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import re
import json
import argparse
import yaml

import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import BERTScorer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
from huggingface_hub import login
from tqdm import tqdm


logging.set_verbosity_error()  # Suppresses all warnings from transformers

# HF login
# login(token=os.getenv("HF_TOKEN"))


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config):
    """Initialize models based on configuration."""
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Select model based on size
    model_name = config['model_name']
    print(f"Loading {model_name} model...")
    
    # Model loading options
    load_options = {
        'torch_dtype': torch.float16 if config['device'] == 'cuda' else torch.float32,
        'low_cpu_mem_usage': True
    }
    
    # Add quantization if specified
    if config.get('quantization', None) == '8bit':
        load_options['load_in_8bit'] = True
    elif config.get('quantization', None) == '4bit':
        load_options['load_in_4bit'] = True
    
    # Add device mapping if using GPU
    if config['device'] == 'cuda':
        load_options['device_map'] = "auto"
    
    # Load tokenizer and model
    llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
    llama_model = AutoModelForCausalLM.from_pretrained(model_name, **load_options)
    
    
    return device, llama_tokenizer, llama_model


def load_data(file_path, max_samples=None):
    """Load and prepare data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    return data


def format_input(question, context, answers, perspective, excerpt):

    if excerpt == '':
        perspective_highlight = ''

    else:
        # Structure for displaying answer spans
        perspective_highlight = f"{perspective}  EXCERPT HIGHLIGHTS:\n" + "\n".join([f"- {s['txt']}" for s in excerpt])
    
    return f"""Question: {question}\n
Context: {context}\n

Current Perspecitve: {perspective}\n

Relevant Answer Excerpts According to current perspective:
{perspective_highlight}\n

Full Answers:
{answers}\n"""


# Define custom function schema for summarization
function_definitions = """[
    {
        "name": "generate_information_summary",
        "description": "Generate information-focused summary from medical answers",
        "parameters": {
            "type": "dict",
            "required": ["summary"],
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise information summary from data shared in answers and from shared answers and from answer excerpts of information perspective"
                }
            }
        }
    },
    {
        "name": "generate_cause_summary",
        "description": "Generate cause-focused summary from medical answers",
        "parameters": {
            "type": "dict",
            "required": ["summary"],
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise summary of root causes or contributing factors shared in answers and from shared answer excerpts of cause perspective"
                }
            }
        }
    },
    {
        "name": "generate_suggestion_summary",
        "description": "Generate treatment suggestions summary from medical answers",
        "parameters": {
            "type": "dict",
            "required": ["summary"],
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise summary of treatment suggestions and recommendations from shared answers and from answer excerpts of information perspective"
                }
            }
        }
    },
    {
        "name": "generate_experience_summary",
        "description": "Generate personal experience summary from medical answers",
        "parameters": {
            "type": "dict",
            "required": ["summary"],
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise summary of personal experiences shared in answer and from answer excerpts of experience perspective"
                }
            }
        }
    }
]"""


system_prompt = """You are a medical summary expert. Analyze the given medical QUESION, CONTEXT, Perspecive-based Labelled Answer Spans, and various ANSWERS given by users to generate four perspective-based summaries using the provided functions. Follow these rules:

Follow these definitions:

1. Question: The user's core medical or health-related query.
2. Context: Background info related to the question (symptoms, experiences, concerns).
3. Answers: Responses from users addressing the question.
4. Labelled Answer Spans: Specific parts of answers categorized by SUGGESTION, CAUSE, or INFORMATION.

Summaries:

1. Information Summary: Focus on factual medical information from authoritative sources
2. Cause Summary: Highlight root causes and contributing factors
3. Suggestion Summary: Summarize treatment recommendations and suggestions
4. Experience Summary: Capture personal experiences shared in answers

Return ALL FOUR function calls with your generated summaries in EXACTLY this format:
[generate_information_summary(summary="..."), generate_cause_summary(summary="..."), generate_suggestion_summary(summary="..."), generate_experience_summary(summary="...")]"""


def parse_function_calls(text):
    """Parse the function calls from the model output with improved error handling"""
    try:
        # Check if the response contains the expected function calls
        if not "generate_" in text or not "summary=" in text:
            # Try to extract any text that might be a summary
            if "summary" in text.lower():
                # Look for something that might be a summary between quotes
                summary_match = re.search(r'"(.*?)"', text)
                if summary_match:
                    return {"summary": summary_match.group(1)}
            
            # If can't find anything useful, return empty summary
            return {"summary": ""}
        
        # Original regex pattern with more flexibility
        pattern = r'generate_(\w+)_summary\(summary="(.*?)"\)'
        matches = re.findall(pattern, text)
        
        if matches:
            return {f"{typ}": txt for typ, txt in matches}
        
        # Try alternative pattern with single quotes
        pattern = r"generate_(\w+)_summary\(summary='(.*?)'\)"
        matches = re.findall(pattern, text)
        
        if matches:
            return {f"{typ}": txt for typ, txt in matches}
        
        # Try looser pattern
        pattern = r'generate_(\w+)_summary.*?summary="?(.*?)"?\)'
        matches = re.findall(pattern, text)
        
        if matches:
            return {f"{typ}": txt for typ, txt in matches}
        
        # If still no matches, extract some text that might be a summary
        return {"summary": text.split("[/INST]")[-1].strip()}
    
    except Exception as e:
        print(f"Error parsing function calls: {e}")
        return {"summary": ""}

def generate_summary_with_llama(question, context, answers, excerpts, perspective, config, llama_tokenizer, llama_model, device):
    """Generate a summary for a specific perspective using Llama 3."""
    # Set up default excerpt as empty list
    excerpt = []
    
    # Extract spans related to the perspective
    perspective_key = f"{perspective}_GROUP"
    if perspective in excerpts:
        excerpt = excerpts[perspective]
    
    # Prepare input
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_input(question, context, answers, perspective, excerpt)}
    ]

    try:
        # Tokenize the prompt
        tokens = llama_tokenizer.apply_chat_template(messages, return_tensors="pt")
        
        # Create an attention mask (to address the warning)
        attention_mask = torch.ones_like(tokens)
        
        # Move tensors to device
        tokens = tokens.to(device)
        attention_mask = attention_mask.to(device)
        
        # Generate the response
        with torch.no_grad():
            outputs = llama_model.generate(
                tokens,
                attention_mask=attention_mask,
                max_new_tokens=config.get('max_new_tokens', 50000),  # Reduced from 50000 to a more reasonable number
                temperature=config.get('temperature', 0.3),  # Slightly lower temperature for more focused responses
                do_sample=config.get('do_sample', True),
                pad_token_id=llama_tokenizer.eos_token_id
            )
        
        # Decode the response
        generated_text = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the model's response after the last instruction
        response_text = generated_text.split("[/INST]")[-1].strip()
        
        # Parse function calls from response
        parsed_functions = parse_function_calls(response_text)
        
        # Check if we have the specific perspective in the parsed functions
        if perspective.lower() in parsed_functions:
            summary = parsed_functions[perspective.lower()]
        elif "summary" in parsed_functions:
            summary = parsed_functions["summary"]
        else:
            # If we can't find a specific summary, use a reasonable fallback
            summary = "Unable to generate a specific summary for this perspective."
            
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"Error occurred while generating {perspective} summary."
    

def calculate_bert_score(generated, reference):
    """Calculate BERT similarity score between generated and reference summaries."""
    if not generated or not reference:
        return 0.0
    
    scorer = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=False)
    P, R, F1 = scorer.score([generated.lower()], [reference.lower()])
    return F1.mean().item()


def calculate_bleu(generated, reference):
    """Calculate BLEU score between generated and reference summaries."""
    if not generated or not reference:
        return 0.0
    
    # Tokenize sentences
    reference_tokens = reference.lower().split()
    generated_tokens = generated.lower().split()
    
    # Calculate BLEU score
    smoothie = SmoothingFunction().method1
    try:
        return corpus_bleu([reference_tokens], [generated_tokens], smoothing_function=smoothie)
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        return 0.0

def process_dataset(file_path, output_path, config, models_dict):
    """Process an entire dataset and evaluate the results."""
    device, llama_tokenizer, llama_model = models_dict.values()
    data = load_data(file_path, config.get('max_samples'))
    results = []
    perspectives = ['INFORMATION', 'CAUSE', 'SUGGESTION', 'EXPERIENCE']
    
    for item in tqdm(data, desc=f"Processing {os.path.basename(file_path)}"):
        result_item = {
            'uri': item.get('uri', ''),
            'question': item.get('question', ''),
            'context': item.get('context', ''),
        }
        
        bleu_scores = {}
        bert_scores = {}
        generated_summaries = {}

        # Process only one entry per dataset initially for debugging
        # Check which perspectives are available in the data
        available_perspectives = []
        for perspective in perspectives:
            summary_key = f"{perspective}_SUMMARY"
            if summary_key in item.get('labelled_summaries', {}):
                available_perspectives.append(perspective)
        
        # If no perspectives are explicitly available, try the first one
        if not available_perspectives and perspectives:
            available_perspectives = [perspectives[0]]
        
        for perspective in available_perspectives:
            # Get reference summary
            reference_summary_key = f"{perspective}_SUMMARY"
            reference_summary = item.get('labelled_summaries', {}).get(reference_summary_key, '')
            
            # Generate summary
            generated_summary = generate_summary_with_llama(
                item.get('question', ''), 
                item.get('context', ''),
                item.get('answers', []),
                item.get('labelled_answer_spans', {}),
                perspective,
                config,
                llama_tokenizer,
                llama_model,
                device
            )
            
            # Only calculate metrics if we have a valid generated summary
            if generated_summary and not generated_summary.startswith("Error"):
                # Calculate metrics
                bleu = calculate_bleu(generated_summary, reference_summary)
                bert = calculate_bert_score(generated_summary, reference_summary)
            else:
                bleu = 0.0
                bert = 0.0
            
            # Store results
            bleu_scores[perspective] = bleu
            bert_scores[perspective] = bert
            generated_summaries[perspective] = generated_summary
            
            # Store reference summaries
            result_item[f"{perspective}_REFERENCE"] = reference_summary
            result_item[f"{perspective}_GENERATED"] = generated_summary
            result_item[f"{perspective}_BLEU"] = bleu
            result_item[f"{perspective}_BERT"] = bert
        
        # Calculate average scores only if we have scores
        if bleu_scores:
            result_item['AVG_BLEU'] = np.mean(list(bleu_scores.values()))
        else:
            result_item['AVG_BLEU'] = 0.0
            
        if bert_scores:
            result_item['AVG_BERT'] = np.mean(list(bert_scores.values()))
        else:
            result_item['AVG_BERT'] = 0.0
        
        results.append(result_item)
        
        # Process just one item per dataset initially for debugging
        # break
    
    # Save detailed results
    output_file = os.path.join(output_path, f"{os.path.basename(file_path).split('.')[0]}_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Calculate and save aggregate metrics
    metrics = {
        'dataset': os.path.basename(file_path),
        'model': f"{config['model_name']}",
        'samples': len(results)
    }
    
    for perspective in perspectives:
        perspective_bleu_values = [r.get(f"{perspective}_BLEU", 0.0) for r in results if f"{perspective}_BLEU" in r]
        perspective_bert_values = [r.get(f"{perspective}_BERT", 0.0) for r in results if f"{perspective}_BERT" in r]
        
        if perspective_bleu_values:
            metrics[f"{perspective}_BLEU_AVG"] = np.mean(perspective_bleu_values)
        else:
            metrics[f"{perspective}_BLEU_AVG"] = 0.0
            
        if perspective_bert_values:
            metrics[f"{perspective}_BERT_AVG"] = np.mean(perspective_bert_values)
        else:
            metrics[f"{perspective}_BERT_AVG"] = 0.0

    # Calculate overall averages
    all_bleu_values = [r.get('AVG_BLEU', 0.0) for r in results]
    all_bert_values = [r.get('AVG_BERT', 0.0) for r in results]

    if all_bleu_values:
        metrics['OVERALL_BLEU_AVG'] = np.mean(all_bleu_values)
    else:
        metrics['OVERALL_BLEU_AVG'] = 0.0
        
    if all_bert_values:
        metrics['OVERALL_BERT_AVG'] = np.mean(all_bert_values)
    else:
        metrics['OVERALL_BERT_AVG'] = 0.0
    
    metrics_file = os.path.join(output_path, f"{os.path.basename(file_path).split('.')[0]}_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    # Parse command line argument for config file
    parser = argparse.ArgumentParser(description='Process healthcare summarization with config file')
    parser.add_argument('--config', type=str, required=False, default='./config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory if it doesn't exist
    os.makedirs(config['output_path'], exist_ok=True)
    
    # Setup models
    models_dict = {
        'device': None,
        'llama_tokenizer': None,
        'llama_model': None,
    }
    
    device, llama_tokenizer, llama_model = setup_model(config)
    models_dict['device'] = device
    models_dict['llama_tokenizer'] = llama_tokenizer
    models_dict['llama_model'] = llama_model
    
    # Save the configuration used
    with open(os.path.join(config['output_path'], 'run_config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Process each dataset
    all_metrics = []
    for dataset in config['datasets']:
        file_path = os.path.join(config['data_path'], dataset)
        if os.path.exists(file_path):
            print(f"\nProcessing {dataset}...")
            metrics = process_dataset(file_path, config['output_path'], config, models_dict)
            all_metrics.append(metrics)
        else:
            print(f"File not found: {file_path}")
    
    # Create a summary report
    summary_df = pd.DataFrame(all_metrics)
    summary_file = os.path.join(config['output_path'], "summary_report.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nEvaluation complete. Results saved to: {config['output_path']}")
    print("\nSummary of results:")
    print(summary_df[['dataset', 'samples', 'OVERALL_BLEU_AVG', 'OVERALL_BERT_AVG']])

if __name__ == "__main__":
    main()