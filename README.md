# Bert_Filter

Use BertForSequenceClassifcation to classify text.

## Build your dataset
{ cat pos.txt; cat neg.txt; } | python build_dataset.py len_train > train.txt<br>
shuf train.txt 

## Train and valuate

sh run_main_predict.sh

## Predict with slid windows
python predict_slid_windows.py
