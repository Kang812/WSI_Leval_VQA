import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms as transforms
from PIL import ImageFile
from torch.optim import lr_scheduler
from tqdm import tqdm
import copy
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch.utils.tensorboard import SummaryWriter

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Whole_Slide_DataSet(Dataset):
    def __init__(self, df_path = "", transform = None, label_seq = ""):
        self.df = pd.read_csv(df_path)
        self.image_paths = self.df['image_path'].to_list()
        self.labels = self.df['label'].to_list()
        self.transform = transform
        self.label_seq = label_seq
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        en_label = self.labels[index]

        if self.transform:
            img = self.transform(image = img)['image']
        
        return img, en_label
    
def classifier_dataloader(df_path = "", transform = None, batch_size = 8, shuffle = True, label_seq = ""):
    dataset = Whole_Slide_DataSet(df_path = df_path, transform = transform, label_seq = label_seq)
    dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader

def score_function(real, pred):
    f1 = f1_score(real, pred, average="macro")
    acc = accuracy_score(real, pred)
    recall = recall_score(real, pred, average="macro")
    precision = precision_score(real, pred, average="macro")
    
    score_dict = dict()
    score_dict['f1_score'] = f1
    score_dict['accuracy'] = acc
    score_dict['recall'] = recall
    score_dict['precision'] = precision
    
    return score_dict

def train_classifer(model, dataloaders, dataset_sizes, epochs, criterion, optimizer, scheduler, device, output_path):

    model = model.to(device)    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = 0.0
    writer = SummaryWriter()

    for i in range(epochs):
        print('Epoch [%d/%d]' % (i + 1, epochs))
        print('-' * 10)

        train_pred = []
        train_y = []

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            pbar = tqdm(dataloaders[phase], unit='unit', unit_scale=True)

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                pbar.update(1)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                train_pred += preds.detach().cpu().numpy().tolist()
                train_y += labels.detach().cpu().numpy().tolist()
        
            if phase == 'train':
                scheduler.step()
        
            scores = score_function(train_y, train_pred)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = scores['accuracy']
            epoch_recall = scores['recall']
            epoch_f1score = scores['f1_score']
            epochs_precision = scores['precision']
            
            pbar.close()

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Recall: {epoch_recall:.4f} Precision: {epochs_precision:.4f} F1 Score: {epoch_f1score:.4f}')
            print()
            
            writer.add_scalar("Loss/%s" % (phase), epoch_loss, i + 1)
            writer.add_scalar("Accuracy/%s" % (phase), epoch_acc, i + 1)
            writer.add_scalar("Recall/%s" % (phase), epoch_recall, i + 1)
            writer.add_scalar("Precision/%s" % (phase), epochs_precision, i + 1)
            writer.add_scalar("F1score/%s" % (phase), epoch_f1score, i + 1)


            if phase == 'val' and epoch_f1score > best_score:
                best_score = epoch_f1score
                best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f'\nBest val f1 score: {best_score:4f}')
    
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), output_path)


def trainer(model, train_dataframe_path, valid_dataframe_path,
            epochs, train_batch_size , valid_batch_size, num_classes,
            output_path, device):
    
    train_transforms = A.Compose([
        A.LongestMaxSize(512, interpolation=cv2.INTER_NEAREST),
        A.PadIfNeeded(min_height=512, min_width=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2()])
    
    val_transforms= A.Compose([
        A.LongestMaxSize(512, interpolation=cv2.INTER_NEAREST),
        A.PadIfNeeded(min_height=512, min_width=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2()])
    
    label_seq = dict()
    for i in range(num_classes):
        label_seq[str(i)] = i
    
    dataset_sizes = dict()
    train_dataset = Whole_Slide_DataSet(df_path = train_dataframe_path, transform = train_transforms, label_seq = label_seq)
    val_dataset = Whole_Slide_DataSet(df_path = valid_dataframe_path, transform = val_transforms, label_seq = label_seq)

    dataset_sizes['train'] = len(train_dataset)
    dataset_sizes['val'] = len(val_dataset)

    data_loaders = dict()
    train_dataloader = classifier_dataloader(df_path = train_dataframe_path, transform = train_transforms, 
                                    batch_size = train_batch_size, shuffle = True, label_seq = label_seq)
    
    val_dataloader = classifier_dataloader(df_path = valid_dataframe_path, transform = val_transforms, 
                                           batch_size = valid_batch_size, shuffle = False, label_seq = label_seq)
    
    data_loaders['train'] = train_dataloader
    data_loaders['val'] = val_dataloader

    ## model training
    epochs = epochs
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    train_classifer(model, data_loaders, dataset_sizes, epochs, criterion, optimizer, scheduler, device, output_path)

def infer(model, image_path, device):
    
    val_transforms= A.Compose([
        A.LongestMaxSize(512, interpolation=cv2.INTER_NEAREST),
        A.PadIfNeeded(min_height=512, min_width=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2()])
    model.to(device)
    model.eval()

    img = cv2.imread(image_path)
    img = val_transforms(image = img)['image']
    img = img.unsqueeze(0)

    with torch.no_grad():
        out = model(img.to(device))
        label_idx = torch.argmax(out, dim=1)
        label_idx = label_idx.detach().cpu().numpy().tolist()[0]
    
    m = nn.Softmax(dim=1)
    output = m(out)
    return label_idx, output