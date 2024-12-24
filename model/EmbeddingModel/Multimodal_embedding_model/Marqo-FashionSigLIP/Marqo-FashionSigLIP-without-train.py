import torch
import os
import torch.nn.functional as F
import random
import pandas as pd

from transformers import AutoModel, AutoProcessor
from PIL import Image

def cosine_similarity(image_features):
    # L2 정규화
    image_features = F.normalize(image_features, p=2, dim=1)  # 각 행(벡터)을 정규화

    # Cosine Similarity 계산 (정규화된 벡터 간의 행렬 곱셈)
    cosine_sim_matrix = image_features @ image_features.T

    return cosine_sim_matrix

def with_all_images(model, processor):
    image_root_url = "/root/25th-conference-EvenT/dataset/Musinsa_dataset"
    image_folders = os.listdir(image_root_url)
    image_folders = [os.path.join(image_root_url, folder) for folder in image_folders]

    image_paths = []
    for folder in image_folders:
        image_path = os.listdir(folder)
        # "random" shuffle the image_path
        random.shuffle(image_path)
        image_path = [os.path.join(folder, path) for path in image_path[:10]]
        image_paths.extend(image_path)

    images = [Image.open(image_path) for image_path in image_paths]
    text =  ['Casual', 'Minimal', 'Street', 'Girlish', 'Workwear', 'Chic', 
            'Gorpcore', 'Classic', 'Sporty', 'Romantic', 'Cityboy', 'Retro', 'Preppy']
    
    processed = processor(text=text, images=images, padding='max_length', return_tensors="pt")
    image_features = model.get_image_features(processed['pixel_values'], normalize=True)

    cs = cosine_similarity(image_features)
    # Make it as 13x13 matrix
    # For result matrix, each grid represents the mean of 10*10 partial matrix of input
    # Round to 4 decimal places
    cs = cs.view(13, 10, 13, 10).mean(dim=(1, 3)).round()
    # save as csv
    pd.DataFrame(cs).to_csv("cosine_similarity.csv", index=False)


def first_trial(model, processor):
    image = [Image.open("/root/25th-conference-EvenT/dataset/Musinsa_dataset/Casual/snap_card_1313739609220618854.jpg"), 
            Image.open("/root/25th-conference-EvenT/dataset/Musinsa_dataset/Casual/snap_card_1319889968174331600.jpg"),
            Image.open("/root/25th-conference-EvenT/dataset/Musinsa_dataset/Sporty/snap_card_1269789102054812102.jpg"),
            Image.open("/root/25th-conference-EvenT/dataset/Musinsa_dataset/Workwear/snap_card_1230706544909193815.jpg")]
    text =  ['Casual', 'Minimal', 'Street', 'Girlish', 'Workwear', 'Chic', 
            'Gorpcore', 'Classic', 'Sporty', 'Romantic', 'Cityboy', 'Retro', 'Preppy']

    processed = processor(images=image, padding='max_length', return_tensors="pt")

    image_features = model.get_image_features(processed['pixel_values'], normalize=True)

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


if __name__ == "__main__":
    model = AutoModel.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)
    processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)

    # first_trial(model, processor)
    with_all_images(model, processor)