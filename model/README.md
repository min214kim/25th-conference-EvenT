# Models

# Task1: Clothing Detection

## Pipeline
We utilized [YOLO](https://arxiv.org/abs/2410.17725) and [EfficientNet](https://arxiv.org/pdf/1905.11946) model. A 1-stage object detection model was fine-tuned on top of a pre-trained YOLO model that was provided by Ultralytics using the K-Fashion dataset. Another model was trained for multi-task image classification using the EfficientNetB3 architecture. It was fine-tuned on top of a EfficientNetB3 model that was pre-trained on the ImageNet dataset. 

## Detailed Information on Models
- [YOLO](YOLO/README.md)
- [EfficientNet](EfficientNet/README.md)

## Dataset
We used [K-fashion dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=51) to fine-tune our model. This dataset provides segmentation information, overall style, bounding boxes for individual clothing items, and detailed clothing attributes. 

### Categories of dataset
This dataset is categorized into four primary classes: 
- 상의(tops)
- 하의(bottoms)
- 원피스(dresses)
- 아우터(outerwear)

Each class is annotated with attributes such as has '세부 카테고리'(sub-category), '컬러'(color), '디테일'(detail), '프린트'(print), '소재'(textile), '사이즈'(size), '기장'(clothing length), and '소매기장'(sleeve length). Additional attributes such as '넥라인'(neckline), and '칼라'(collar) information are included for relevant clothing types.

# Task2: Outfit Of The Day(OOTD) image embedding

## Detailed Information on Models
The detailed information is in [EmbeddingModel](EmbeddingModel/README.md)

## Dataset
We crawled style image from [Musinsa](https://www.musinsa.com/). We utilized 13 different styles (Casual, Chic, Cityboy, Classic, Girlish, Gorpcore, Minimal, Preppy, Retro, Romantic, Sporty, Street, Workwear), and about 2,000 images for each styles. 