import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from model import DREmodel, new_DREmodel
from dataSet import Dataset
from trainer import Trainer
from transformers import AdamW, get_linear_schedule_with_warmup

device = torch.device('cuda:0')
train_path = 'data/new_train.json'
val_path = 'data/new_dev.json'
test_path = 'data/new_test.json'
BATCH_SIZE = 4
TEST_SIZE = 12
max_seq_len = 256

def train(model, train_data, val_data, test_data, model_name):
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_data, shuffle=True, batch_size=TEST_SIZE)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=TEST_SIZE)
    
    #训练
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate
                      eps=1e-8  # args.adam_epsilon
                      )

    # Number of training epochs
    epochs = 15

    #scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)


    #多标签分类损失函数
    criterion = nn.BCELoss()
    
    trainer = Trainer(model=model,
                       trainloader=train_dataloader,
                       valloader=val_dataloader, 
                       testloader=test_dataloader, 
                       optimizer=optimizer,
                       device=device,
                       criterion=criterion,
                       epoch=epochs,
                       model_name=model_name
                      )
    trainer.train()

train_data = Dataset(train_path, max_seq_len)
val_data = Dataset(val_path, max_seq_len)
test_data = Dataset(test_path, max_seq_len)
model_1 = DREmodel().to(device)
#model_2 = new_DREmodel().to(device)
train(model_1, train_data, val_data, test_data, model_name='baseline')
#train(model_2, train_data, val_data, test_data, model_name='new')

