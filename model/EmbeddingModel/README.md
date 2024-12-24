# Outfit Of The Day (OOTD) image embedding

## Abstract
We developed a model to generate image embeddings for a recommendation service that displays style-similar images. Using style images crawled from [Musinsa](https://www.musinsa.com/), we leveraged pretrained models like ResNet and EfficientNet. To improve training performance, we preprocessed images with a [clothing detection model](../YOLO/README.md) to focus only on the clothings. However, the performance was still bad, we decided to use multimodal embedding model and fine-tune to our dataset.

## Necessity
Our service aims to display images similar to the styles that users prefer. To achieve this, we built a vector database (vectorDB) to store image embeddings. Since the database requires inputs in the form of image embeddings, we developed this model to generate and store those embeddings effectively.

## Methods

### Fine-tune Imagenet Pretrained Model
Initially, we attempted to implement contrastive learning using full OOTD images with Imagenet-pretrained models such as ResNet, EfficientNet, and ResNeXt. However, we observed that the images contained too much unnecessary information, such as background details, which overshadowed the relevant style information. As a result, the training process was ineffective, with loss values stagnating and accuracy not improving.
To address this, we decided to preprocess the images using a clothing detection model. First, we extracted the bounding boxes (bboxes) for the clothing items. Then, we determined the bounding box for the entire human, and used it as an input to the image embedding model for better feature representation.
We trained our model multiple times, but its performance was not satisfactory. We suspect this is because the person is always positioned at the center of the image, making most images look too similar. Even after trying an ImageNet-pretrained model, we struggled to reach our desired performance level.

### Multimodal embedding model
After exploring multiple trials on pretrained model on Imagenet, we decided to use multimodal embedding model called [Marqo-FashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP). Since this model outputs a 768-dimensional vector-which was more than we needed-we added a linear layer to reduce it to 128 dimensions.