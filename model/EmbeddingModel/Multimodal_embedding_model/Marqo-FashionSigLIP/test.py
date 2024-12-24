from model import EmbeddingModel

import torch
import os
import torch.nn.functional as F
import random
import pandas as pd

from PIL import Image
from torchvision import transforms

model_path = "/root/25th-conference-EvenT/model/EmbeddingModel/Multimodal_embedding_model/Marqo-FashionSigLIP/best_result_200_1e2_8_32_margin_1.pth"
model = EmbeddingModel()
model.load_state_dict(torch.load(model_path))

def with_small_sample():
    images = [Image.open("/root/25th-conference-EvenT/dataset/Musinsa_dataset_cropped/Casual/snap_card_1313740852494523114.jpg"), 
            Image.open("/root/25th-conference-EvenT/dataset/Musinsa_dataset_cropped/Casual/snap_card_1319876212664957871.jpg"),
            Image.open("/root/25th-conference-EvenT/dataset/Musinsa_dataset_cropped/Sporty/snap_card_1267740799923069346.jpg"),
            Image.open("/root/25th-conference-EvenT/dataset/Musinsa_dataset_cropped/Workwear/snap_card_1230705656151379919.jpg")]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
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

def cosine_similarity(image_features):
    # L2 정규화
    image_features = F.normalize(image_features, p=2, dim=1)  # 각 행(벡터)을 정규화

    # Cosine Similarity 계산 (정규화된 벡터 간의 행렬 곱셈)
    cosine_sim_matrix = image_features @ image_features.T

    return cosine_sim_matrix

def with_all_images():
    image_root_url = "/root/25th-conference-EvenT/dataset/Musinsa_dataset"
    image_folders = os.listdir(image_root_url)
    image_folders = [os.path.join(image_root_url, folder) for folder in image_folders]

    image_paths = []
    for folder in image_folders:
        image_path = os.listdir(folder)
        # "random" shuffle the image_path
        random.shuffle(image_path)
        image_path = [os.path.join(folder, path) for path in image_path[:100]]
        image_paths.extend(image_path)

    images = [Image.open(image_path) for image_path in image_paths]
    
    image_features = model(images)

    cs = cosine_similarity(image_features)
    # Make it as 13x13 matrix
    # For result matrix, each grid represents the mean of 10*10 partial matrix of input
    cs = cs.view(13, 100, 13, 100).mean(dim=(1, 3))
    print(cs)
    # print maximum column index of each row
    print(cs.argmax(dim=1))

    # save as csv
    pd.DataFrame(cs).to_csv("cosine_similarity.csv", index=False)

if __name__ == "__main__":
    with_all_images()