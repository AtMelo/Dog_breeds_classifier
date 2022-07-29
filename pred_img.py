import os

import torch
import torchvision
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import yaml
from PIL import Image

# Load parameters
with open('cnfg.yaml') as f:
    configs = yaml.safe_load(f)

test_dir = configs['test_dir']

mean_all_data = tuple(configs['mean_all_data'])
std_all_data = tuple(configs['std_all_data'])
folder2label = {'n02086240': 'Shih-Tzu',
                'n02087394': 'Rhodesian ridgeback',
                'n02088364': 'Beagle',
                'n02089973': 'English foxhound',
                'n02093754': 'Australian terrier',
                'n02096294': 'Border terrier',
                'n02099601': 'Golden retriever',
                'n02105641': 'Old English sheepdog',
                'n02111889': 'Samoyed',
                'n02115641': 'Dingo'}

model_nn = torchvision.models.resnet34(num_classes=10)
model_nn.load_state_dict(
    torch.load('./dog_resnet34.pt', map_location='cpu')['state_dict']
)
model_nn.eval()

valid_tfms = tt.Compose(
    [tt.Resize((400, 500)),
     tt.ToTensor(),
     tt.Normalize(mean_all_data, std_all_data)
     ]
)
test_folder = ImageFolder(test_dir, valid_tfms)


def predict_img(model, img, transform, classes):
    img = transform(Image.open(img))
    img = img.unsqueeze(0)
    out = model(img)
    _, preds = torch.max(out, dim=1)
    SM = torch.nn.Softmax(dim=1)
    out_probability = torch.max(SM(out), dim=1)
    out_probability = out_probability.values.item()
    return classes[preds], out_probability


path_dir = os.getcwd()

predicted_class, probability = predict_img(
    model_nn,
    path_dir,
    valid_tfms,
    test_folder.classes
)
mess_predict = f'Predict with probability ({probability:.2f}):' \
               f' {folder2label[predicted_class]}'
print(mess_predict)
