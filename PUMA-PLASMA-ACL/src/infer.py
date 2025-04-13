import json
import argparse 
from transformers.file_utils import PushToHubMixin
from transformers import GPT2Tokenizer, GPT2Model,GPT2LMHeadModel,AutoModelForSeq2SeqLM,AutoTokenizer
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType,PeftModel
import sys
sys.path.insert(0, './') 
from train_dataloader import * 
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from torch.optim import AdamW
import numpy as np
import os
import pandas as pd
device = 'cuda'
if __name__=="__main__":

##########################################################################
# Prepare Parser
##########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', default="../data/test_preprocessed.json")
    parser.add_argument('--model_file', type=str, required=False)
    parser.add_argument('--batch_size_test', type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt_name", type=str, default="adapter_epoch=10_valid_loss=32.818")
    
    args = parser.parse_args()
    
    TEST_BATCH_SIZE = args.batch_size_test
    with open(args.test_file, 'r') as json_file:
        test_data = json.load(json_file)
    EPOCHS = args.num_epochs
    
    model_name = "google/flan-t5-base"

    foundation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    peft_model_path = f"{args.ckpt_dir}/{args.ckpt_name}"
    
    loaded_model = PeftModel.from_pretrained(
    foundation_model,  # The base model to be used for prefix tuning
    peft_model_path,   # The path where the trained Peft model is saved
    is_trainable=False  # Indicates that the loaded model should not be trainable
    ).to(device)
    
    test_dataset = CustomDataset(test_data,tokenizer)
    test_dataloader = test_create_dataloader(test_dataset, TEST_BATCH_SIZE)

    perspectives = []
    predictions = []
    actuals = []
    inputs = []
            
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader)):
            input_text = batch["input_ids"].to(device)
            input_attention = batch["attention_mask"].to(device)
            outputs =  loaded_model.generate(input_ids=input_text,attention_mask=input_attention,num_beams=5, max_new_tokens=500,temperature=0.9, repetition_penalty=1.2)
         
            output_text = tokenizer.decode(outputs[0]).replace('<pad>','').replace('</s>','').strip(" ")

            perspective = test_data[step]['Perspective']
            perspectives.append(perspective)

            predicted = [output_text]
            predictions.append(predicted)

            actual = test_data[step]['Summary']
            actuals.append(actual)

            entry_input = [test_data[step]['answers']]
            inputs.append(entry_input)
            
            data = {"perspective": perspective, "predicted": predicted, "actual": actual, "input": entry_input}
            
            print()
            for key, value in data.items():
                print(f"{key}: {value}")
            print()

    df = pd.DataFrame(columns = ['Perspective', 'Prediction', 'Actual', 'Input'])
    df['Perspective'] = perspectives
    df['Prediction'] = predictions
    df['Actual'] = actuals
    df['Input'] = inputs

    save_path = './generated/generated_result.csv'
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_csv(save_path, index=False)
    print(f"generated predictions saved to: {save_path}!")









