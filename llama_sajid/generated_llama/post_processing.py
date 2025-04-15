import pandas as pd

def extract_relevant_text(text):
    """Extracts the main summary substring from text by removing everything before 
    'Content to summarize: ' and everything after 'Question:'. as llama gives prompts as well"""
    
    start_marker = "Content to summarize: "
    end_marker = "Question:"
    
    if start_marker in text and end_marker in text:
        start_index = text.index(start_marker) + len(start_marker)
        end_index = text.index(end_marker)
        return text[start_index:end_index].strip()
    else: 
        return text


df = pd.read_csv('/home/iiitd/Sajid/NLP_Project/sajid/generated_llama/generated_resultcsv')
df['Prediction'] = df['Prediction'].apply(extract_relevant_text)
df.to_csv('generated_result_final.csv', index=False)
print("Processing complete. The modified CSV has been saved as 'generated_result_final.csv'.")
