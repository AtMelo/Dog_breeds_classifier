import telebot
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


with open('token.txt', 'r') as f:
    TOKEN = f.read()

bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start'])
def start(message):
    mess = f'Hello {message.from_user.first_name}. I can help you classify' \
           f' 10 classes of dog breed. Run /help to understand what to do.'
    bot.send_message(message.chat.id, mess)


@bot.message_handler(commands=['help'])
def start(message):
    mess = f'I can classify this breeds: Shih-Tzu, ' \
           f'Rhodesian ridgeback, Beagle, English foxhound, ' \
           f'Australian terrier, Border terrier, Golden retriever, ' \
           f'Old English sheepdog, Samoyed and Dingo'
    bot.send_message(message.chat.id, mess)
    mess2 = 'What you need to do? Just attach a photo of one of the dogs' \
            ' and compress the photo'
    bot.send_message(message.chat.id, mess2)


# @bot.message_handler(commands=['info'])
# def user_info(message):
#     bot.send_message(message.chat.id, message)


@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    try:

        file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        path_dir = '/home/anton/PycharmProjects/Dog_breeds_classifier' \
                   '/Load_imgs/'
        src = path_dir + file_info.file_path
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.reply_to(message, "I need to think...")

        predicted_class, probability = predict_img(
            model_nn,
            src,
            valid_tfms,
            test_folder.classes
        )
        mess_predict = f'Predict with probability ({probability:.2f}):' \
                       f' {folder2label[predicted_class]}'
        bot.send_message(message.chat.id, mess_predict)

    except Exception as e:
        bot.reply_to(message, e)


bot.polling(none_stop=True)
