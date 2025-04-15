import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
sys.path.insert(0, './')
from train_dataloader import CustomDataset, test_create_dataloader
from tqdm.auto import tqdm
import torch
import os
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', default="/home/iiitd/Sajid/NLP_Project/sajid/llama_sajid/data/test_preprocessed.json")
    parser.add_argument('--batch_size_test', type=int, default=4)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt_name", type=str, default="adapter_epoch=4_valid_loss=13.6436")
    
    args = parser.parse_args()
    
    with open(args.test_file, 'r') as json_file:
        test_data = json.load(json_file)
    
   
    model_name = "KidIkaros/tiny-llama3.2-instruct"
    foundation_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16  
    ).to(device)
    print(f"Base model loaded: {model_name} on device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    peft_model_path = f"{args.ckpt_dir}/{args.ckpt_name}"
    loaded_model = PeftModel.from_pretrained(
        foundation_model,
        peft_model_path,
        is_trainable=False
    ).to(device)

    
    print(f"PEFT adapter model loaded from: {peft_model_path} on device: {device}")
    
    test_dataset = CustomDataset(test_data, tokenizer)
    test_dataloader = test_create_dataloader(test_dataset, args.batch_size_test)

    perspectives, predictions, actuals, inputs = [], [], [], []
            
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader)):
            input_text = batch["input_ids"].to(device)
            input_attention = batch["attention_mask"].to(device)
            
           
            outputs = loaded_model.generate(
                input_ids=input_text,
                attention_mask=input_attention,
                num_beams=1,         
                do_sample=True,      
                temperature=0.9,
                max_new_tokens=256,  
                repetition_penalty=1.2
            )
            
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            perspective = test_data[step]['Perspective']
            perspectives.append(perspective)
            predictions.append([output_text])
            actuals.append(test_data[step]['Summary'])
            inputs.append([test_data[step]['answers']])
            
            data = {"perspective": perspective, "predicted": [output_text], "actual": test_data[step]['Summary'], "input": [test_data[step]['answers']]}
            print()
            for key, value in data.items():
                print(f"{key}: {value}")
            print()

    df = pd.DataFrame({
        'Perspective': perspectives,
        'Prediction': predictions,
        'Actual': actuals,
        'Input': inputs
    })

    save_path = './generated_llama/generated_result.csv'
    save_dir = os.path.dirname(save_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df.to_csv(save_path, index=False)
    print(f"Generated predictions saved to: {save_path}!")
