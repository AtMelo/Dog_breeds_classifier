import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torchvision.transforms as transforms
import yaml


# Load parameters
with open('cnfg.yaml') as f:
    configs = yaml.safe_load(f)

test_dir = configs['test_dir']

mean_all_data = tuple(configs['mean_all_data'])
std_all_data = tuple(configs['std_all_data'])
folder2label = configs['folder2label']

batch_size = configs['batch_size']


def model_check(model, test_loader, dev='cpu'):
    with torch.no_grad():
        model.eval()
        test_acc = 0
        for data, label in test_loader:
            data, label = data.to(dev), label.to(dev)

            output = model(data)

            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(label.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            test_acc += accuracy.item() * data.size(0)

        test_acc = test_acc / len(test_loader.dataset)
        return test_acc


# Load the model for testing
model_nn = torchvision.models.resnet34(num_classes=10)
model_nn.load_state_dict(torch.load('./dog_resnet34.pt')['state_dict'])
model_nn.eval()
criterion = torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

valid_tfms = tt.Compose(
        [transforms.Resize((400, 500)),
         tt.ToTensor(),
         tt.Normalize(mean_all_data, std_all_data)
         ]
    )
test_ds = ImageFolder(test_dir, valid_tfms)
test_dataset = DataLoader(test_ds, batch_size, shuffle=True)

test_ac = model_check(model_nn.to(device),
                      test_dataset,
                      dev=device)

print(f'The model has achieved an accuracy of {100 * test_ac:.2f}%'
      f' on the test dataset')
