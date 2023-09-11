# Bert_Filter

Use BertForSequenceClassifcation to classify text.

## Build your dataset
{ cat pos.txt; cat neg.txt; } | python build_dataset.py len_train > train.txt
shuf train.txt 

## Train and valuate

sh run_main_predict.sh
