import argparse

import torch
from torch.nn import BCELoss
from multi_classify_model import BertForMultiLabelSequenceClassification
from pytorch_pretrained_bert import BertAdam
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm, trange
from numpy import mean


from dataset_utils import Artical_Dataset
from settings import bert_path, train_200_path

# python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="10.131.2.33" --master_port=23456 rddp.py
#
# python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr="10.131.2.33" --master_port=23456 rddp.py

# nproc_per_node 每个节点有几个显卡
# nnodes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ranglage Pytorch Distributed Train')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    # 不用设置该参数，会根据nproc_per_node和nnodes自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='', help='url used to set up distributed training')
    # 一张显卡能8个样本
    # gradient accumulation 8
    # 4个节点
    parser.add_argument("--train_batch_size",
                        default=256,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs",
                        default=500,
                        type=int,
                        help="Total number of training epochs to perform.")

    args = parser.parse_args()

    model = BertForMultiLabelSequenceClassification.from_pretrained(bert_path, num_labels=4)

    train_dataset=Artical_Dataset(train_200_path)

    num_train_optimization_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        backend = 'NCCL'
        dev = torch.device("cuda", args.local_rank)
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend)
        model = model.to(dev)
        model = DDP(model)
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=5e-5,
                         warmup=0.1,
                         t_total=num_train_optimization_steps)

    loss_fn=BCELoss()
    model.train()
    loss_epoch = dict()

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        accumulation_steps=0
        accumulation_loss=[]
        loss_batch = []
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            accumulation_steps += 1
            x, y = tuple(t.to(dev) for t in batch)
            output = model(x)
            y=y.type_as(output)
            loss=loss_fn(output.view(-1, 1), y.view(-1, 1))
            accumulation_loss.append(loss.item())
            loss.backward()
            if accumulation_steps%args.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()
                mean_accu_loss = mean(accumulation_loss)
                print('accumulation batch loss', mean_accu_loss)
                loss_batch.append(mean_accu_loss)
                accumulation_loss=[]
        mean_epoch_loss=mean(loss_batch)
        loss_epoch[epoch] = loss_batch
    torch.save(model,
               f'/home/pb064/Ranglage/Models/FinBERT_L-12_H-768_A-12_pytorch/small_fine_tuning_classify.pt')
    torch.save(loss_epoch,
               '/home/pb064/Ranglage/Models/FinBERT_L-12_H-768_A-12_pytorch/small_fine_tuning_classify_train_loss.pt')





