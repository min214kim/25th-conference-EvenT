# Outfit of the Day Image Classification Model

## Abstract
By using EfficientNetB3 as the base model architecture, we used it to train this model to perform multi-task image classification from the cropped images obtained from our YOLO object detection model. This model will reshape the dimensions of the images as well as normalize the input image, which is then put through our model to output the specific class of that article of clothing, the length, as well as the color. 

## Necessity
We developed a unified model designed to perform multi-class image classification across 12 distinct tasks, focusing on analyzing clothing articles. The model's primary function is to classify each article into one of four main categories: Outerwear, Upperwear, Lowerwear, or One-Piece garments. Additionally, it handles three subtasks for each main category: identifying the specific subcategory (e.g., T-shirt, jacket, skirt), determining the length attribute (e.g., short, long, knee-length), and recognizing the dominant color of the item.

## Models
We utilized the EfficientNetB3 model that receives input images of size (300 x 300). 