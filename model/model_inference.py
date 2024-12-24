from model import EfficientNetMultiTask, EmbeddingModel

import os
import json
import torch
import argparse

from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Define combined label mappings for all clothing types and tasks
label_to_number = {
    '아우터': {
        'task1': {
            "코트": 0, "재킷": 1, "점퍼": 2, "패딩": 3, "가디건": 4, "짚업": 5
        },
        'task2': {
            "크롭": 0, "노멀": 1, "하프": 1, "맥시": 2, "롱": 2
        },
        'task3': {
            '블루': 0, '블랙': 1, '골드': 2, '실버': 3, '화이트': 4, '와인': 5,
            '그린': 6, '라벤더': 7, '옐로우': 8, '브라운': 9, '핑크': 10,
            '카키': 11, '네온': 12, '레드': 13, '그레이': 14, '민트': 15,
            '네이비': 16, '스카이블루': 17, '퍼플': 18, '오렌지': 19, '베이지': 20
        }
    },
    '하의': {
        'task1': {
            "청바지": 0, "팬츠": 1, "스커트": 2, "조거팬츠": 3
        },
        'task2': {
            "미니": 0, "미디": 1, "니렝스": 1, "맥시": 2, "발목": 2
        },
        'task3': {
            '블루': 0, '블랙': 1, '골드': 2, '실버': 3, '화이트': 4, '와인': 5,
            '그린': 6, '라벤더': 7, '옐로우': 8, '브라운': 9, '핑크': 10,
            '카키': 11, '네온': 12, '레드': 13, '그레이': 14, '민트': 15,
            '네이비': 16, '스카이블루': 17, '퍼플': 18, '오렌지': 19, '베이지': 20
        }
    },
    '상의': {
        'task1': {
            "티셔트": 0, "니트웨어": 1, "셔츠": 2, "후드티": 3
        },
        'task2': {
            "크롭": 0, "노멀": 1, "롱": 1
        },
        'task3': {
            '블루': 0, '블랙': 1, '골드': 2, '실버': 3, '화이트': 4, '와인': 5,
            '그린': 6, '라벤더': 7, '옐로우': 8, '브라운': 9, '핑크': 10,
            '카키': 11, '네온': 12, '레드': 13, '그레이': 14, '민트': 15,
            '네이비': 16, '스카이블루': 17, '퍼플': 18, '오렌지': 19, '베이지': 20
        }
    },
    '원피스': {
        'task1': {
            "드레스": 0
        },
        'task2': {
            "미니": 0, "미디": 1, "니렝스": 1, "맥시": 2
        },
        'task3': {
            '블루': 0, '블랙': 1, '골드': 2, '실버': 3, '화이트': 4, '와인': 5,
            '그린': 6, '라벤더': 7, '옐로우': 8, '브라운': 9, '핑크': 10,
            '카키': 11, '네온': 12, '레드': 13, '그레이': 14, '민트': 15,
            '네이비': 16, '스카이블루': 17, '퍼플': 18, '오렌지': 19, '베이지': 20
        }
    }
}

number_to_label = {
    '아우터': {
        'task1': {
            0: "코트", 1: "재킷", 2: "점퍼", 3: "패딩", 4: "가디건", 5: "짚업"
        },
        'task2': {
            0: "크롭", 1: "노멀", 2: "맥시"
        },
        'task3': {
            0: '블루', 1: '블랙', 2: '골드', 3: '실버', 4: '화이트', 5: '와인',
            6: '그린', 7: '라벤더', 8: '옐로우', 9: '브라운', 10: '핑크',
            11: '카키', 12: '네온', 13: '레드', 14: '그레이', 15: '민트',
            16: '네이비', 17: '스카이블루', 18: '퍼플', 19: '오렌지', 20: '베이지'
        }
    },
    '하의': {
        'task1': {
            0: "청바지", 1: "팬츠", 2: "스커트", 3: "조거팬츠"
        },
        'task2': {
            0: "미니", 1: "미디", 2: "맥시"
        },
        'task3': {
            0: '블루', 1: '블랙', 2: '골드', 3: '실버', 4: '화이트', 5: '와인',
            6: '그린', 7: '라벤더', 8: '옐로우', 9: '브라운', 10: '핑크',
            11: '카키', 12: '네온', 13: '레드', 14: '그레이', 15: '민트',
            16: '네이비', 17: '스카이블루', 18: '퍼플', 19: '오렌지', 20: '베이지'
        }
    },
    '상의': {
        'task1': {
            0: "티셔트", 1: "니트웨어", 2: "셔츠", 3: "후드티"
        },
        'task2': {
            0: "크롭", 1: "노멀"
        },
        'task3': {
            0: '블루', 1: '블랙', 2: '골드', 3: '실버', 4: '화이트', 5: '와인',
            6: '그린', 7: '라벤더', 8: '옐로우', 9: '브라운', 10: '핑크',
            11: '카키', 12: '네온', 13: '레드', 14: '그레이', 15: '민트',
            16: '네이비', 17: '스카이블루', 18: '퍼플', 19: '오렌지', 20: '베이지'
        }
    },
    '원피스': {
        'task1': {
            0: "드레스"
        },
        'task2': {
            0: "미니", 1: "미디", 2: "맥시"
        },
        'task3': {
            0: '블루', 1: '블랙', 2: '골드', 3: '실버', 4: '화이트', 5: '와인',
            6: '그린', 7: '라벤더', 8: '옐로우', 9: '브라운', 10: '핑크',
            11: '카키', 12: '네온', 13: '레드', 14: '그레이', 15: '민트',
            16: '네이비', 17: '스카이블루', 18: '퍼플', 19: '오렌지', 20: '베이지'
        }
    }
}

def load_efficientNet_model(model_path):
    # Define number of classes per task dynamically
    num_classes_per_task = {
        clothing_type: {task: len(mapping) for task, mapping in tasks.items()}
        for clothing_type, tasks in label_to_number.items()
    }

    # Load the model
    model = EfficientNetMultiTask(num_classes_per_task)
    model.load_state_dict(torch.load(model_path))

    return model

def load_Embedding_model(model_path):
    model = EmbeddingModel()
    model.load_state_dict(torch.load(model_path))

    return model

def transform_image(image):
    # Transforms the image to a tensor for input to the model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    tensor = transform(image).unsqueeze(0)  

    return tensor

def get_task_labels(model, image, image_class):
    image_classes = ['아우터', '하의', '원피스', '상의']
    image_class_str = image_classes[image_class]

    number_to_label_task = number_to_label[image_class_str]

    results = []
    for i in range(1, 4):
        # For each task, get the prediction
        task = f"task{i}"
        result = model(image, f"{image_class_str}_{task}").argmax(dim=1).item()
        result = number_to_label_task[task][result]
        results.append(result)
    
    return results

def final_inference_per_image(YOLO_model, EfficientNet_model, Embedding_model, image_root, image_folder, image_file, imageID):
    s3url_root = "https://even-t.s3.ap-northeast-2.amazonaws.com/Musinsa_to_s3"
    imageID = int(imageID)
    image_path = f"{image_root}/images/{image_folder}/{image_file}"
    # Load the image
    image = Image.open(image_path)

    # Get embedding vector for the image
    embedding_vector = Embedding_model([image])

    # Get the YOLO results
    YOLO_result = YOLO_model(image, verbose=False)

    # Get the attributes of each box using EfficientNet
    attributes = []

    # Make the directory for saving the cropped image
    cropped_image_path = f"{image_root}/cropped_image/{image_folder}/{image_file.split('.')[0]}"
    if not os.path.exists(cropped_image_path):
        os.makedirs(cropped_image_path)

    # Get the attributes of each box
    idx = 1
    for box in YOLO_result[0].boxes:
        # Get the coordinates and class of the box
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        image_class = int(box.cls.item())
        
        # Crop the image using the coordinates of the box
        cropped_image = image.crop((x1, y1, x2, y2))
        
        # Transform the cropped image
        cropped_tensor = transform_image(cropped_image)

        # Get attributes of the cropped image
        task_labels = get_task_labels(EfficientNet_model, cropped_tensor, image_class)

        # Make the attribute dictionary and append to the list
        attribute = {"categoryName": task_labels[0],
                     "attributes": {"color": task_labels[2], 
                                    "length": task_labels[1]},
                     "s3Url": f"{s3url_root}/cropped_image/{image_folder}/{image_file}/{idx}.jpg"}

        # Save the cropped image
        cropped_image.save(f"{cropped_image_path}/{idx}.jpg")
        attributes.append(attribute)
        idx += 1


    # TODO
    # Upload to s3, and get s3url
    s3url = f"{s3url_root}/images/{image_folder}/{image_file}"

    # Make final json file
    mongo_json = {}
    mongo_json["clothesId"] = imageID
    mongo_json["fulls3url"] = s3url
    mongo_json["categories"] = attributes
    mongo_json["vector"] = embedding_vector.tolist()

    vectorDB_json = {}
    vectors = [{
        "id": imageID,
        "values": embedding_vector.tolist(),
        "metadata": {
            "clothesId": imageID,
            "fulls3url": s3url
        }
    }]
    vectorDB_json["vectors"] = vectors
    vectorDB_json["namespace"] = ""

    # Save the final json file
    Mongo_json_root = "/root/25th-conference-EvenT/dataset/Musinsa_to_s3/MongoDB_json"
    vectorDB_json_root = "/root/25th-conference-EvenT/dataset/Musinsa_to_s3/vectorDB_json"

    json_file = image_file.split(".")[0]
    
    Mongo_json_path = f"{Mongo_json_root}/{image_folder}/{json_file}.json"
    vectorDB_json_path = f"{vectorDB_json_root}/{image_folder}/{json_file}.json"

    # Make the directory if it does not exist
    if not os.path.exists(f"{Mongo_json_root}/{image_folder}"):
        os.makedirs(f"{Mongo_json_root}/{image_folder}")
    if not os.path.exists(f"{vectorDB_json_root}/{image_folder}"):
        os.makedirs(f"{vectorDB_json_root}/{image_folder}")
    
    # Save the final json file in korean
    with open(Mongo_json_path, 'w', encoding='utf-8') as f:
        json.dump(mongo_json, f, ensure_ascii=False, indent=4)
    with open(vectorDB_json_path, 'w', encoding='utf-8') as f:
        json.dump(vectorDB_json, f, ensure_ascii=False, indent=4)

def final_inference():
    # Load the YOLO model
    YOLO_model_path = "/root/25th-conference-EvenT/model/YOLO/2stage/yolo11m_fine_tuning_val.pt"
    YOLO_model = YOLO(YOLO_model_path)

    # Load the EfficientNet model
    EfficientNet_model_path = "/root/25th-conference-EvenT/model/EfficientNet/checkpoints/best_model.pth"
    EfficientNet_model = load_efficientNet_model(EfficientNet_model_path)

    # Load the embedding model
    Embedding_model_path = "/root/25th-conference-EvenT/model/EmbeddingModel/Multimodal_embedding_model/Marqo-FashionSigLIP/best_result_200_1e2_8_32_margin_1.pth"
    Embedding_model = load_Embedding_model(Embedding_model_path)

    # Image root folder setting
    image_root = "/root/25th-conference-EvenT/dataset/Musinsa_to_s3"
    image_folders = os.listdir(f"{image_root}/images")

    imageID = 1
    for image_folder in image_folders:
        image_files = os.listdir(f"{image_root}/images/{image_folder}")

        for image_file in tqdm(image_files):
            if not image_file.endswith(".jpg"):
                continue

            final_inference_per_image(YOLO_model, EfficientNet_model, Embedding_model, image_root, image_folder, image_file, imageID)
            imageID += 1

if __name__ == "__main__":
    # image_root = "/root/25th-conference-EvenT/dataset/Musinsa_to_s3"
    # image_folder = "Casual"
    # image_file = "snap_card_1313738332830481199.jpg"

    # final_inference_per_image(YOLO_model, EfficientNet_model, Embedding_model, image_root, image_folder, image_file, 1)
    final_inference()