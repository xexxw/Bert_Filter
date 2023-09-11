import os
import json
from transformers import AutoTokenizer
from huggingface_hub import notebook_login
from tqdm import tqdm


def preprocess(tokenizer, text, max_length):
    
    return tokenizer.encode_plus(text, 
                                 add_special_tokens=True, 
                                 return_attention_mask=True, 
                                 max_length=max_length, 
                                 adding='max_length', 
                                 truncation=True)


def get_token(dataset, max_length, tag):
    notebook_login()
    token_path = os.path.join(dataset, f'{tag}_token.txt')
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            data = json.load(f)
            labels = data['labels']
            input_ids = data["input_ids"]
            attention_masks = data['attention_mask']
    else:
        texts = []
        labels = []
        # filename_ls = os.listdir(input_path)
        # filename = filename_ls[epoch % len(filename_ls)]
        # load_file = os.path.join(input_path,filename)
        filename = os.path.join(dataset, f'{tag}.txt')
        with open(filename, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                if len(parts) != 2:continue
                texts.append(parts[0])
                labels.append(int(parts[-1]))

        # inputs = list(map(preprocess, texts))
        inputs = [preprocess(tokenizer, text, max_length) for text in tqdm(texts)]
        input_ids = [x['input_ids'] for x in inputs]
        attention_masks = [x['attention_mask'] for x in inputs]
        
        data = {'input_ids':input_ids, 'attention_mask':attention_masks, 'labels':labels}
        with open(token_path, 'w') as f:
            json.dump(data, f)
            
        del texts
        del inputs
    del data

    return input_ids, attention_masks, labels
