import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import numpy as np
from utils import get_token
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, f1_score, \
    confusion_matrix
from huggingface_hub import notebook_login

notebook_login()

def get_args_from_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="", required=False, help="数据集")
    parser.add_argument("--load_path", type=str, default="checkpoint", required=False, help="加载模型路径")
    parser.add_argument("--max_length",default=512,type =int,required=False,help="最大长度")
    args = parser.parse_args()
    return args  

def evaluate(y_pred, y_true):
    print("Precision: ", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("F1 score: ", f1_score(y_true, y_pred))
    print("Confusion Matrix:", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))


# 使用已经训练好的模型对测试集进行预测
args = get_args_from_parser()
input_ids, attention_masks, labels = get_token(args.data_path, tag='test', max_length=args.max_length)
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)
labels = torch.tensor(labels)
datasets = TensorDataset(input_ids, attention_masks, labels)


model = AutoModelForSequenceClassification.from_pretrained('hfl/chinese-roberta-wwm-ext')

model_list = os.listdir(args.load_path)

test_dataloader = DataLoader(datasets, batch_size=1800)
num_training_steps = len(model_list) * len(test_dataloader)
progress_bar = tqdm(range(num_training_steps))


for model_id in model_list:
    model_path = os.path.join(args.load_path, model_id)
        
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
            output = model(input_ids, attention_mask=attention_masks, labels=labels)               
            prediction = output[1]
            y_pred_batch = torch.max(prediction, 1)[1]
            y_true.extend(labels.tolist())
            y_pred.extend(y_pred_batch.tolist())
            progress_bar.update(1)
        
        print("model_id:", model_id)
        evaluate(y_pred, y_true)
    
    with open(args.data_path, "r+", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].strip() + "\t" + model_id + "\t"+ str(y_pred[i]) + "\n"
        f.seek(0)
        f.writelines(lines)