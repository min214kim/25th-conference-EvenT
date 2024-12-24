from model import EmbeddingModel

import torch

from PIL import Image
from torchvision import transforms

model_path = "/root/25th-conference-EvenT/model/EmbeddingModel/resnet-wo_pretrain.pth"
model = EmbeddingModel()
model.load_state_dict(torch.load(model_path))


images = [Image.open("/root/25th-conference-EvenT/dataset/Musinsa_dataset_cropped/Casual/snap_card_1313740852494523114.jpg"), 
          Image.open("/root/25th-conference-EvenT/dataset/Musinsa_dataset_cropped/Casual/snap_card_1319876212664957871.jpg"),
          Image.open("/root/25th-conference-EvenT/dataset/Musinsa_dataset_cropped/Sporty/snap_card_1267740799923069346.jpg"),
          Image.open("/root/25th-conference-EvenT/dataset/Musinsa_dataset_cropped/Workwear/snap_card_1230705656151379919.jpg")]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

results = []
for image in images:
    image = transform(image).unsqueeze(0)
    result = model(image)
    results.append(result[0].squeeze(0).tolist())
    print(result[0].squeeze(0).shape)

image_features = torch.tensor(results)

cos_sim01 = torch.nn.functional.cosine_similarity(image_features[0], image_features[1], dim=0)
cos_sim02 = torch.nn.functional.cosine_similarity(image_features[0], image_features[2], dim=0)
cos_sim03 = torch.nn.functional.cosine_similarity(image_features[0], image_features[3], dim=0)
cos_sim12 = torch.nn.functional.cosine_similarity(image_features[1], image_features[2], dim=0)
cos_sim13 = torch.nn.functional.cosine_similarity(image_features[1], image_features[3], dim=0)
cos_sim23 = torch.nn.functional.cosine_similarity(image_features[2], image_features[3], dim=0)


print("Cosine similarity between image 0 and image 1:", cos_sim01)
print("Cosine similarity between image 0 and image 2:", cos_sim02)
print("Cosine similarity between image 0 and image 3:", cos_sim03)
print("Cosine similarity between image 1 and image 2:", cos_sim12)
print("Cosine similarity between image 1 and image 3:", cos_sim13)
print("Cosine similarity between image 2 and image 3:", cos_sim23)