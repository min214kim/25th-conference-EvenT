import argparse
import torch
import os
import torch.nn as nn

from model import EfficientNetMultiTask
from dataloader import get_dataloaders
from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate EfficientNetB7 on Multiple Tasks")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory containing train, val, and test folders")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for DataLoaders")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], required=True, help="Mode: train or evaluate")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument('--output_dir', type=str, default=".", help="Directory to save model checkpoints and logs")

    args = parser.parse_args()

    # Define combined label mappings for all clothing types and tasks
    label_mappings = {
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

    # Define number of classes per task dynamically
    num_classes_per_task = {
        clothing_type: {task: len(mapping) for task, mapping in tasks.items()}
        for clothing_type, tasks in label_mappings.items()
    }

    # Initialize model
    model = EfficientNetMultiTask(num_classes_per_task=num_classes_per_task)

    # If multiple GPUs are available, wrap the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    print('Count of using GPUs:', torch.cuda.device_count())   

    # Get DataLoaders (train, val, test)
    dataloaders = get_dataloaders(args.data_dir, label_mappings, args.batch_size, args.num_workers)

    # Define optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    if args.mode == 'train':
        # Train the model with validation
        checkpoint_dir = "/root/25th-conference-EvenT/model/EfficientNet/checkpoints"
        best_model = train_model(
            model, dataloaders['train'], dataloaders['val'],
            optimizer, criterion, args.device, args.num_epochs,
            checkpoint_dir=checkpoint_dir
        )

        # Save the best model
        output_path = f"{checkpoint_dir}/best_model.pth"
        # If using DataParallel, best_model might be wrapped, so always save state_dict from model.module if needed
        if isinstance(best_model, torch.nn.DataParallel):
            torch.save(best_model.module.state_dict(), output_path)
        else:
            torch.save(best_model.state_dict(), output_path)

        print(f"Model saved to {output_path}")

    elif args.mode == 'evaluate':
        # Evaluate the model
        evaluate_model(model, dataloaders['test'], args.device)
