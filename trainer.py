import torch
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, trainloader, valloader, testloader, optimizer, device, criterion, epoch=10, model_name='baseline'):
        '''
        :param model: 模型
        :param trainloader: 训练集加载
        :param testloader: 测试集加载
        :param optimizer: 优化器
        :param device: 设备
        :param criterion: 损失函数
        :param epoch: epoch
        '''
        self.device = device
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = epoch
        self.model_name = model_name

    def _train_epoch(self):
        self.model.train()
        avg_loss = 0
        for step, batch in enumerate(self.trainloader):
         
            self.model.zero_grad()
            label = batch['labels'].squeeze().to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            y = self.model(input_ids, attention_mask)
            loss = self.criterion(y, label)
            loss.backward()
            self.optimizer.step()
            # 记录loss
            if(step % 500 == 0):
                print("itration: ", step, ", loss=", loss.cpu().detach().numpy())
                avg_loss += loss
        avg_loss = avg_loss / 3
        return avg_loss

    def _val_epoch(self, itr, sigma = 0.6):
        self.model.eval()
        labels = []
        preds = []
        for step, batch in enumerate(self.valloader):
            label = batch['labels'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            pred = self.model(input_ids, attention_mask)
            pred = pred.cpu().detach().numpy()
            m, n = pred.shape
            for i in range(m):
                for j in range(n):
                    if(pred[i][j] > sigma):
                        pred[i][j] = 1
                    else:
                        pred[i][j] = 0
            labels.extend(list(label.cpu().detach().numpy()))
            preds.extend(list(pred))
        
        F1 = metrics.f1_score(labels, preds, average='micro')
            
        # 统计acc
        print("val dataset: epoch=", itr, ", F1 score = ", F1 * 100, "%")
        
    def _test_epoch(self, itr, sigma = 0.6):
        self.model.eval()
        labels = []
        preds = []
        for step, batch in enumerate(self.testloader):
            label = batch['labels'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            pred = self.model(input_ids, attention_mask)
            pred = pred.cpu().detach().numpy()
            m, n = pred.shape
            for i in range(m):
                for j in range(n):
                    if(pred[i][j] > sigma):
                        pred[i][j] = 1
                    else:
                        pred[i][j] = 0
            labels.extend(list(label.cpu().detach().numpy()))
            preds.extend(list(pred))
        
        F1 = metrics.f1_score(labels, preds, average='micro')
            
        # 统计acc
        print("test dataset: epoch=", itr, ", F1 score = ", F1 * 100, "%")

    def train(self):
        losses = []
        for itr in range(self.epoch):
            losses.append(self._train_epoch())
            if(self.model_name == 'baseline'):
                torch.save(self.model,'/home/models/baseline/'+str(itr)+'.pth')
            if(self.model_name == 'new'):
                torch.save(self.model,'/home/models/new/'+str(itr)+'.pth')
            self._val_epoch(itr)
            self._test_epoch(itr)
        
        plt.plot(losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        if(self.model_name == 'baseline'):
            plt.savefig('losses_1.jpg', dpi=100)
        if(self.model_name == 'new'):
            plt.savefig('losses_2.jpg', dpi=100)
