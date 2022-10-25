from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix, multilabel_confusion_matrix, f1_score, classification_report
from tqdm import tqdm
import transformers
from transformers import AutoConfig,AutoModel,BertTokenizer
from transformers import BertModel, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import random
import argparse
import sys
import re 
import os 


class BertNetwork(nn.Module):    
    def __init__(self,device): #preTrainedModel    
        super(BertNetwork, self).__init__()
        #TODO: Create a rule for the model selection
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.nb_features = self.model.pooler.dense.out_features
        self.drop = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(self.nb_features,3)
        self.device = device
        
        for param in self.model.parameters(): #Freeze BERT weights
            param.requires_grad = False


    def forward(self,input_ids=None,attention_mask=None):
        output = self.model(input_ids,attention_mask,return_dict=True)
        ft = self.drop(output['pooler_output'])
        x = self.fc1(ft) 
        return x


    def train(self,train_loader,epochs,criterion,optimizer,scheduler):
        total = 0.0
        correct = 0.0
        tr_loss = 0.0
        with tqdm(train_loader, unit='batch') as tepoch:
            for input_id,attention_mask,label in tepoch:
        
                input_id = input_id.to(self.device)
                attention_mask = attention_mask.to(self.device)
                label = label.to(self.device)                                
                tepoch.set_description(f'Epoch {epoch + 1}/{epochs}')
                
                optimizer.zero_grad()
                output = self.forward(input_id,attention_mask)    
                loss = criterion(output, label)
                tr_loss += loss.item()
                scores, prediction = torch.max(output, dim=1)           
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total += label.size(0)
                correct += (prediction == label).sum().item()
                accuracy = correct / total                                    
                tepoch.set_postfix(loss=loss.item(),accuracy=100.*accuracy)
            
            torch.cuda.empty_cache()
            print(f'\nFinished Training in {epoch + 1} with loss={loss:.2f} and accuracy={100 * accuracy:.2f}')
        return tr_loss, correct
    
    
    def test(self, testloader):
        total = 0.0
        correct = 0.0
        tst_loss = 0.0
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            with tqdm(testloader, unit='batch') as tepoch:
        
                for input_id,attention_mask,label in tepoch:                
                    input_id = input_id.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    label = label.to(self.device)                
                    y_true.extend(label.cpu().numpy())
                    
                    output = self.forward(input_id,attention_mask)
                    loss = criterion(output, label)
                    tst_loss += loss.item()        
                    scores, prediction = torch.max(output.data, 1)
                    y_pred.extend(prediction.cpu().numpy())
                    
                    total += label.size(0)
                    correct += (prediction == label).sum().item()
                    tepoch.set_postfix(accuracy=100. * (correct / total))
        print(f'\nFinished Testing with accuracy={100 * (correct / total):.2f}')
        return y_true, y_pred, tst_loss, correct



def preprocessing_for_bert(data):
    """Preprocess data for BERT model

    Args:
        data (numpy.ndarray): data to preprocess for BERT model

    Returns:
        torch.Tensor: preprocessed data for BERT model input format 
    """
    input_ids = []
    attention_masks = []
    
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=128,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            return_attention_mask=True      # Return attention mask
        )
        
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
        
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks


def preprocess(text):
    """Preprocess text for model.

    Args:
        text (str): Text to preprocess

    Returns:
        str: Preprocessed text
    """
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'!',' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def plot_the_confusion_matrix(true_labels, predicted_labels):
        encoded_classes = ['Positive', 'Negative', 'Neutral']
        predicted_category = [encoded_classes[np.argmax(x)] for x in predicted_labels]
        true_category = [encoded_classes[x] for x in true_labels]

        x = 0
        for i in range(len(true_category)):
            if true_category[i] == predicted_category[i]:
                x += 1
        print('Accuracy Score = ', x / len(true_category))
        confusion_mat = confusion_matrix(y_true = true_category, y_pred = predicted_category, labels=list(encoded_classes))
        df = pd.DataFrame(confusion_mat, index = list(encoded_classes),columns = list(encoded_classes))

        ax = sns.heatmap(df,cmap='Blues',annot=True)
        plt.title('Heatmap for Bert Model')
        #plt.savefig('heatmap.png')
        plt.show()


def plot_loss(loss_dict):
        """ Plot the mode's loss 

        Args:
            loss_dict (dict): dictionary containing the loss values
        """
        plt.figure(figsize=(10,8))
        plt.semilogy(loss_dict['train_loss_ep'], label='Train')
        plt.semilogy(loss_dict['test_loss_ep'], label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.grid()
        plt.legend()
        plt.title('Training vs Testing loss for 2-fold cross validation')
        plt.show()
  
    
def plot_accuracy(acc_dict):
    """ Plot the mode's accuracy

    Args:
        acc_dict (dict): _description_
    """
    plt.figure(figsize=(10,8))
    plt.semilogy(acc_dict['train_acc_ep'], label='Train')
    plt.semilogy(acc_dict['test_acc_ep'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.grid()
    plt.legend()
    plt.title('Training vs Testing accuracy for 2-fold cross validation')
    plt.show()


def plot_the_class_distr(df,col_name):
        ax = df[col_name].value_counts().plot(kind='bar', figsize=(10, 6), fontsize=13, color='#087E8B')
        ax.set_title('Class distribution (0 = negative, 1 = neutral, 2 = positive)', size=20, pad=30)
        ax.set_ylabel('Sentiment', fontsize=14)

        for i in ax.patches:
            ax.text(i.get_x() + 0.19, i.get_height() + 700, str(round(i.get_height(), 2)), fontsize=15)
        plt.show()



if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs',help='Set the number of epochs', type=int, default=4)
    parser.add_argument('-folds',help='Set the number of folds', type=int, default=2)
    parser.add_argument('-batch',help='Set the batch size number', type=int, default=4)
    parser.add_argument('-model',help='Select the transformer model',default='bert-base-uncased')
    args = parser.parse_args()
    
    num_of_epochs = args.epochs
    num_of_folds = args.folds
    batch_size = args.batch

    
    df = pd.read_csv('Covid_19_tweets.csv')
    sentences = df['OriginalTweet'].values
    labels = df['encoded_sentiment'].values  
    
    for i in range(len(sentences)) :
        sentences[i] = preprocess(sentences[i])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Pytorch running on: {device}')
    
    #TODO: Create a rule for the tokinizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)  
    criterion = nn.CrossEntropyLoss()

    fold = 0
    foldperf={}
    
    kf = StratifiedKFold(n_splits=num_of_folds,shuffle=True,random_state=42)     
    for train_index, test_index in kf.split(sentences, labels):
        print('Fold {}'.format(fold + 1))
        X_train, X_val = sentences[train_index], sentences[test_index]
        y_train, y_val = labels[train_index], labels[test_index]  
        
        train_inputs, train_masks = preprocessing_for_bert(X_train)
        tst_inputs, tst_masks = preprocessing_for_bert(X_val)
        train_labels = torch.tensor(y_train)
        tst_labels = torch.tensor(y_val) 

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        tst_data = TensorDataset(tst_inputs, tst_masks, tst_labels)
        tst_sampler = SequentialSampler(tst_data)
        tst_dataloader = DataLoader(tst_data, sampler=tst_sampler, batch_size=batch_size)
        
        net = BertNetwork(device).to(device) #preTrainedModel        
        optimizer = AdamW(net.parameters(),lr=5e-5,eps=1e-8)
        total_steps = len(train_dataloader) * num_of_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)
        
        history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
        for epoch in range(num_of_epochs):
            
            train_loss, train_correct= net.train(train_dataloader,num_of_epochs,criterion,optimizer,scheduler)
            #true_labels, predicted_labels, test_loss, test_correct = net.test(tst_dataloader)
            _, _, test_loss, test_correct = net.test(tst_dataloader)
        
            train_loss = train_loss / len(train_dataloader.sampler)
            train_acc = train_correct / len(train_dataloader.sampler) * 100
            test_loss = test_loss / len(tst_dataloader.sampler)
            test_acc = test_correct / len(tst_dataloader.sampler) * 100

            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

            '''
            print("Epoch:{}/{}AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc:{:.2f}% AVG Test Acc:{:.2f}%".format(epoch+1,
                    num_of_epochs,
                    train_loss,
                    test_loss,
                    train_acc,
                    test_acc))
            '''
        foldperf['fold{}'.format(fold+1)] = history    
        fold += 1   
    
    #TODO: Save the model 
    
    
    
    testl_f,tl_f,testa_f,ta_f=[],[],[],[]
    for f in range(1,num_of_folds+1):
        
        tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))

        ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
        testa_f.append(np.mean(foldperf['fold{}'.format(f)]['test_acc']))
    print('Performance of {} fold cross validation'.format(num_of_folds))
    print('Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f}'.format(np.mean(tl_f),np.mean(testl_f),np.mean(ta_f),np.mean(testa_f)))
    
    
    diz_ep = {'train_loss_ep':[],'test_loss_ep':[],'train_acc_ep':[],'test_acc_ep':[]}

    for i in range(num_of_epochs):
        diz_ep['train_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_loss'][i] for f in range(num_of_folds)]))
        diz_ep['test_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['test_loss'][i] for f in range(num_of_folds)]))
        diz_ep['train_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_acc'][i] for f in range(num_of_folds)]))
        diz_ep['test_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['test_acc'][i] for f in range(num_of_folds)]))
        
    plot_loss(diz_ep)      
    plot_accuracy(diz_ep)
    
    #TODO: Plot the confusion matrix