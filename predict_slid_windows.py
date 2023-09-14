import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json
import numpy as np
import argparse
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, f1_score, \
    confusion_matrix
from datetime import datetime
# from huggingface_hub import notebook_login

# notebook_login() # 加载model和tokennizer

def get_args_from_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="allcorpus", required=False, help="数据集")
    parser.add_argument("--load_path", type=str, default="checkpoint/2023_09_08_18_20_11", required=False, help="加载模型路径")
    parser.add_argument("--max_length",default=512,type=int,required=False,help="max_length default = 256")
    parser.add_argument("--batch_size",default=64,type=int,required=False,help="batch_size default = 512")
    parser.add_argument("--n_windows",default=4,type=int,required=False,help="n_windows default = 4")
    parser.add_argument("--threshold",default=0.8,type=float,required=False,help="threshold default = 0.8")
    args = parser.parse_args()
    return args  


def evaluate(y_pred, y_true):
    print("Precision: ", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("F1 score: ", f1_score(y_true, y_pred))
    print("Confusion Matrix:", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

# 加载测试集数据
def preprocess(text, max_length, n_windows): 
    windows = []
    if len(text) < max_length:
        windows = [text for _ in range(n_windows)]
    else:
        step = n_windows - 1
        stride = (len(text) - max_length)//step  # 计算窗口之间的滑动距离
        for i in range(n_windows):
            start = i * stride  # 计算当前窗口的起始位置
            end = min(start + max_length, len(text))  # 计算当前窗口的结束位置
            window_text = text[start:end]  # 提取当前窗口的文本
            windows.append(window_text)
    return tokenizer.batch_encode_plus(windows, 
                                 add_special_tokens=True, 
                                 return_attention_mask=True, 
                                 max_length=max_length, 
                                 padding='max_length', 
                                 truncation=True)


args = get_args_from_parser()
current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_inputs = os.path.join(args.dataset, f"val_{args.n_windows}_token.json")  
test_file = os.path.join(args.dataset, "val.txt")  
out_file = os.path.join(args.dataset, f"val_{args.threshold}_{current_time}.txt")
tokenizer = BertTokenizer.from_pretrained('chinese_roberta_wwm_ext')
if os.path.exists(save_inputs):
    with open(save_inputs, 'r') as f:
        data = json.load(f)
        labels = data['labels']
        input_ids = data["input_ids"]
        attention_masks = data['attention_mask']
        
else:
    texts = []
    labels = []
    with open(test_file, 'r') as f: 
        for line in f.readlines():
            parts = line.strip().split('\t')
            if len(parts) != 2:
                texts.append(parts[0])
                labels.append(int(parts[1]))
                continue
            texts.append(parts[0])
            labels.append(int(parts[-1]))

    # inputs = list(map(preprocess, texts))
    inputs = [preprocess(text, args.max_length, args.n_windows) for text in tqdm(texts)]
    input_ids = [x['input_ids'] for x in inputs]
    attention_masks = [x['attention_mask'] for x in inputs]
    
    data = {'input_ids':input_ids, 'attention_mask':attention_masks, 'labels':labels}
    with open(save_inputs, 'w') as f:
        json.dump(data, f)
            
        
    del texts
    del inputs
del data



# 使用已经训练好的模型对测试集进行预测
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)
labels = torch.tensor(labels)
datasets = TensorDataset(input_ids, attention_masks, labels)




model = BertForSequenceClassification.from_pretrained('chinese_roberta_wwm_ext')

model_list = os.listdir(args.load_path)


test_dataloader = DataLoader(datasets, batch_size=args.batch_size)

num_training_steps = len(model_list) * len(test_dataloader)
progress_bar = tqdm(range(num_training_steps))



for model_id in model_list:
    model_path = os.path.join(args.load_path, model_id)  # 替换为您已经训练好的模型的路径
        
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    y_true = []
    y_pred = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration", ascii=True)):                       
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)
            y_pred_windows_list = []
            for i in range(args.n_windows):
                input_ids_window = input_ids[:, i, :].squeeze()
                attention_masks_window = attention_masks[:, i, :].squeeze()
                output = model(input_ids_window, attention_mask=attention_masks_window, labels=labels)
                
                logits = output.logits
                probabilities = torch.softmax(logits, dim=1)
                positive_probs = probabilities[:, 0]
                y_pred_windows_list.append(positive_probs)


            y_pred_windows = torch.stack(y_pred_windows_list, dim=1)

            y_true.extend(labels.tolist())

            for i in range(y_pred_windows.size(0)):
                row_prediction = []
                for j in range(y_pred_windows.size(1)):
                    if y_pred_windows[i, j] >= args.threshold:                    
                        row_prediction.append(0)
                    else:
                        row_prediction.append(1)
                

                y_pred.append(min(row_prediction))

            progress_bar.update(1)


        print("model_id:", model_id)
        evaluate(y_pred, y_true)
    

    
    with open(test_file, "r+", encoding="utf-8") as f_in, open(out_file, "w", encoding="utf-8") as f_out:
        lines = f_in.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].strip() + "\t" + model_id + "\t"+ str(y_pred[i]) + "\n"
        f_out.writelines(lines)