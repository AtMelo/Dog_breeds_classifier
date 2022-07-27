import pandas as pd

import torchvision
import torch.nn as nn
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
import random
import torchvision.transforms as transforms
import yaml
from utils import *

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

# Load parameters
with open('cnfg.yaml') as f:
    configs = yaml.safe_load(f)

path_dataset = configs['path_dataset']
train_dir = configs['train_dir']
test_dir = configs['test_dir']

mean_all_data = tuple(configs['mean_all_data'])
std_all_data = tuple(configs['std_all_data'])
folder2label = configs['folder2label']

batch_size = configs['batch_size']


def model_init(num_classes):
    model = torchvision.models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, num_classes, bias=True)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    print("Tunable Layers: ")
    for (name, param) in model.named_parameters():
        if param.requires_grad:
            print(f'{name} -> {param.requires_grad}')

    return model


def train(model,
          criterion,
          optimizer,
          train_loader,
          val_loader,
          save_location,
          early_stop=3,
          n_epochs=20,
          print_every=2, to_dev='cpu'):
    # Initializing some variables
    valid_loss_min = np.Inf
    stop_count = 0
    valid_max_acc = 0
    history = []
    model.epochs = 0

    # Loop starts here
    for epoch in range(n_epochs):

        train_loss = 0
        valid_loss = 0

        train_acc = 0
        valid_acc = 0

        model.train()
        ii = 0

        for data, label in train_loader:
            ii += 1
            data, label = data.to(to_dev), label.to(to_dev)
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # Track train loss by multiplying average loss by number
            # of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output,
                                dim=1)  # first output gives the max value in
            # the row(not what we want), second output gives index
            # of the highest val

            correct_tensor = pred.eq(
                label.data.view_as(pred)
            )  # using the index of the predicted outcome above, torch.eq()
            # will check prediction index against label index to see if
            # prediction is correct(returns 1 if correct, 0 if not)

            accuracy = torch.mean(
                correct_tensor.type(
                    torch.FloatTensor
                )
            )  # tensor must be float to calc average

            train_acc += accuracy.item() * data.size(0)
            if ii % 10 == 0:
                print(
                    f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}'
                    f'% complete.')

        model.epochs += 1
        with torch.no_grad():
            model.eval()

            for data, label in val_loader:
                data, label = data.to(to_dev), label.to(to_dev)

                output = model(data)
                loss = criterion(output, label)
                valid_loss += loss.item() * data.size(0)

                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(label.data.view_as(pred))
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                valid_acc += accuracy.item() * data.size(0)

            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(val_loader.dataset)

            train_acc = train_acc / len(train_loader.dataset)
            valid_acc = valid_acc / len(val_loader.dataset)

            history.append([train_loss, valid_loss, train_acc, valid_acc])

            if (epoch + 1) % print_every == 0:
                print(
                    f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \t'
                    f'Validation Loss: {valid_loss:.4f}')
                print(
                    f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t '
                    f'Validation Accuracy: {100 * valid_acc:.2f}%')

            if valid_loss < valid_loss_min:

                torch.save({
                    'state_dict': model.state_dict(),
                }, save_location)
                stop_count = 0
                valid_loss_min = valid_loss
                valid_best_acc = valid_acc
                best_epoch = epoch

            else:
                stop_count += 1

                # Below is the case where we handle the early stop case
                if stop_count >= early_stop:
                    print(
                        f'\nEarly Stopping Total epochs: {epoch}. '
                        f'Best epoch: {best_epoch} with loss:'
                        f' {valid_loss_min:.2f} and'
                        f' acc: {100 * valid_acc:.2f}%')
                    model.load_state_dict(
                        torch.load(save_location)['state_dict'])
                    model.optimizer = optimizer
                    history = pd.DataFrame(history,
                                           columns=['train_loss', 'valid_loss',
                                                    'train_acc', 'valid_acc'])
                    return model, history

    model.optimizer = optimizer
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} '
        f'and acc: {100 * valid_acc:.2f}%')

    history = pd.DataFrame(history,
                           columns=['train_loss', 'valid_loss', 'train_acc',
                                    'valid_acc'])
    return model, history


def main():
    train_tfms = tt.Compose(
        [
            transforms.Resize((400, 500)),
            tt.RandomCrop((400, 500),
                          padding=4,
                          padding_mode='reflect'),
            tt.RandomHorizontalFlip(),
            tt.ToTensor(),
            tt.Normalize(mean_all_data, std_all_data, inplace=True)
        ]
    )

    valid_tfms = tt.Compose(
        [transforms.Resize((400, 500)),
         tt.RandomCrop((400, 500)),
         tt.ToTensor(),
         tt.Normalize(mean_all_data, std_all_data)
         ]
    )
    train_folder = ImageFolder(train_dir, train_tfms)
    test_folder = ImageFolder(test_dir, valid_tfms)

    val_size = int(0.15 * len(train_folder))
    train_size = int(len(train_folder) - val_size)

    data_train, data_val = random_split(train_folder, [train_size, val_size])
    train_dl = DataLoader(data_train, batch_size, shuffle=True)
    val_dl = DataLoader(data_val, batch_size, shuffle=True)

    model_nn = model_init(num_classes=len(train_folder.classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_nn.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_nn.parameters(),
                                 lr=1e-3,
                                 betas=(0.9, 0.999)
                                 )
    model_nn, history = train(
        model_nn,
        criterion,
        optimizer,
        train_dl,
        val_dl,
        save_location='./dog_resnet34.pt',
        early_stop=3,
        n_epochs=30,
        print_every=2,
        to_dev=device)

    return model_nn, history


if __name__ == '__main__':
    main()
