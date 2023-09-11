import numpy as np
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import logging
import random
from pathlib import Path 
from tqdm import tqdm, trange
import time
from datetime import timedelta
from utils import get_token
from huggingface_hub import notebook_login



def get_args_from_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",default=None,type=Path,required=True,help="save model checkpoint")
    parser.add_argument("--data_path", default=None, type=Path, required=True, help="dataset path")
    parser.add_argument("--load_state_dir",default=None,type=Path,required=False,help="model state dict load")
    parser.add_argument("--eval_data_path",default=None,type=Path,help="eval_data_path")
    parser.add_argument("--save_inputs",default=["output"],type=Path,required=False,help="save_inputs")
    parser.add_argument("--max_length",default=512,type =int,required=False,help="max_length default = 512")
    parser.add_argument('--eval_step',type=int,default=500,help="eval step default = 500")
    parser.add_argument("--num_train_epochs",default=10,type=int,required=True,help="model train epochs default 10")
    parser.add_argument("--batch_size",default=64,type =int,required=True,help="batch_size default = 64")
    parser.add_argument("--gradient_accumulation_steps",default=1,type = int,required=False,help="gradient_accumulation_steps default = 1")
    parser.add_argument("--learning_rate",default=5e-5,required=False,type=float,help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion",default=0.1,type=float,required=False,help="Proportion of training to perform linear learning rate warmup for. ""E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',type=int,default=42,required=False,help="random seed for initialization")
    parser.add_argument("--DDP",action='store_true',required=False,help="use DDP to train model")
    parser.add_argument("--fp16",action='store_true',required=False,help="use fp16 to train model")
    args = parser.parse_args()
    return args  


def main():
    notebook_login()
    rank = 0
    args = get_args_from_parser()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('args:{}'.format(args))

    
    if args.DDP:
        import torch.distributed as dist
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        rank = dist.get_rank()
        logger.info("train in DDP mode, device: %s ,ues gpu num is %d" % (device,n_gpu))
        logger.info(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) train worker starting...")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        logger.info("train in DP mode,device : {} ,n_gpu : {}, 16-bits:{}".format(device,n_gpu,args.fp16))


    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    if rank == 0 and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    
    model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext")
    if args.load_state_dir != None:
        load_state_path=args.load_state_dir
        logger.info("load state dict from %s " % (args.load_state_dir))
        model.load_state_dict(torch.load(load_state_path,map_location=torch.device("cpu")))
    
    model.to(device)
    
    if args.DDP:
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    elif n_gpu > 1:
        gpus = [i for i in range(n_gpu)]
        model = torch.nn.DataParallel(model,device_ids=gpus, output_device=gpus[0])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    global_step = 0

    nb_tr_steps = 1

    logger.info("***** Running dataloder *****")
    # 将数据转换为 PyTorch 张量并构建 DataLoader

    input_ids, attention_masks, labels = get_token(args.data_path, max_length=args.max_length, tag="Train")
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)
    datasets = TensorDataset(input_ids, attention_masks, labels)
        
    if args.DDP:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(datasets)
    else:
        from torch.utils.data import RandomSampler
        train_sampler = RandomSampler(datasets)
    train_dataloader = DataLoader(datasets,sampler=train_sampler,batch_size=args.batch_size)
    num_training_steps = args.num_train_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    logger.info("***** Running training *****")
    logger.info("  Batch size = %d" % (args.batch_size))
    logger.info("  num_train_epochs = %d" % (args.num_train_epochs))
    logger.info("  dataloader len = %d" % (len(train_dataloader)))
    logger.info("  Training_steps = %d"  % (num_training_steps))

    
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
            
            output = model(input_ids = batch[0],
                            attention_mask = batch[1],
                            labels = batch[2])
            
            loss = output.loss
            if n_gpu > 1:
                loss = loss.mean() 
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            
            mean_loss = loss * args.gradient_accumulation_steps / nb_tr_steps
            progress_bar.update(1)
            
            
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0 and (global_step + 1) % args.eval_step == 0:
                    result = {}
                    logger.info("  epoch {} step {} loss {} \n".format(epoch, step, loss ))
                    result['global_step'] = global_step
                    result['loss'] = mean_loss
                    output_eval_file = os.path.join(args.output_dir, "log.txt")
                    with open(output_eval_file, "a") as writer:
                        logger.info("***** Eval results *****")
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))
                    model_name = "step_{}_loss_{}".format(global_step,mean_loss)
                    output_config_file = os.path.join(args.output_dir,"config.json")
                    logger.info("** ** * Saving fine-tuned model ** ** * ")
                    model_to_save = model.module if hasattr(model, 'module') else model
                    
                    output_model_file = os.path.join(args.output_dir, model_name)
                    model_to_save.config.to_json_file(output_config_file)
                    torch.save(model_to_save.state_dict(), output_model_file)
        if rank == 0:
            model_name = "epoch_{}_loss_{}".format(epoch,mean_loss)
            logger.info("** ** * Saving fine-tuned model ** ** * ")
            model_to_save = model.module if hasattr(model, 'module') else model
            output_config_file = os.path.join(args.output_dir,"config.json")
            output_model_file = os.path.join(args.output_dir, model_name)
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)


if __name__ == "__main__":
    main()
