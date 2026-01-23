# EvenT

<div align="center">
<h3>24-2 YBIGTA Conference</h2>
<img src="asset/EvenT_logo.png" alt="logo" width="300"/>
</div>

> EvenT is a outfit recommendation service based on digital closet

## Table of Contents
- [EvenT](#event)
  - [Table of Contents](#table-of-contents)
  - [📍 Problem Definition](#-problem-definition)
  - [📍 Tech Stack](#-tech-stack)
    - [AI \& Data](#ai--data)
    - [Backend \& Database](#backend--database)
    - [Frontend \& Design](#frontend--design)
  - [📍 Results and Key Features](#-results-and-key-features)
    - [1. Style Swipe](#1-style-swipe)
    - [2. Personal Closet DB](#2-personal-closet-db)
  - [Limitations and Future Work](#limitations-and-future-work)
    - [1. Closet DB Construction via Order History](#1-closet-db-construction-via-order-history)
    - [2. EfficientNet Training Optimization](#2-efficientnet-training-optimization)
    - [3. Purchase Recommendations](#3-purchase-recommendations)
  - [Team Composition](#team-composition)

## 📍 Problem Definition
- **Challenges**
    1. **Inefficient Consumption & Environmental Impact**: Purchasing repetitive styles or leaving clothes unworn leads to textile waste and environmental pollution.
    2. **Decision Fatigue**: People experience fatigue during the shopping process due to a lack of clear purchasing criteria.
- **Solutions from EvenT**
    1. **Enhancing "Closet Meta-cognition"**: Helping users clearly visualize and understand their current inventory to prevent redundant purchases.
    2. **Style Profiling**: Analyzing frequently worn items and user preferences to define a clear, personalized fashion identity.

## 📍 Tech Stack

### AI & Data
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces?style=for-the-badge)
![Selenium](https://img.shields.io/badge/-Selenium-%2343B02A?style=for-the-badge&logo=selenium&logoColor=white)
- **Object Detection**: YOLO v11
- **Attribute Classification**: EfficientNet-B3
- **Style Embedding**: FashionSigLIP
- **Data**: K-Fashion Dataset, Musinsa Snap Crawling (Crawled 18,000+ style images using Selenium and BeautifulSoup)

### Backend & Database
![Spring Boot](https://img.shields.io/badge/Spring_Boot-6DB33F?style=for-the-badge&logo=springboot&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=for-the-badge&logo=mongodb&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=for-the-badge)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
- **Database**: MongoDB 
- **Vector Search**: Pinecone
- **Cloud Storage**: AWS S3 (Image hosting)

### Frontend & Design
![Flutter](https://img.shields.io/badge/Flutter-%2302569B.svg?style=for-the-badge&logo=Flutter&logoColor=white)
![Figma](https://img.shields.io/badge/figma-%23F24E1E.svg?style=for-the-badge&logo=figma&logoColor=white)



## 📍 Results and Key Features

### 1. Style Swipe
<p align="center">
  <img src="asset/Swipe.png" alt="Swipe" height="300"/>
</p>

Users can select "Like/Dislike/Save" while viewing style images to discover their personal preferences.

To achieve this, we vectorized style images into a **VectorDB** and stored a **Preference Vector** that represents the user's preferred style. When a user uses the swipe service, the system extracts images from the VectorDB that are most similar to the user's preference vector. For style image vectorization, we used an **Embedding Model** [[README]](model/EmbeddingModel/README.md) trained with an additional custom layer.

---

### 2. Personal Closet DB

EvenT digitizes the user's actual clothing into a database. Two methods are used to build this DB: **Indirect Digitization** and **Direct Digitization**.

<p align="center">
    <img src="asset/indirect_DB.png" alt="indirect_DB" height="300"/>
    <img src="asset/direct_DB.png" alt="direct_DB" height="300"/>
</p>

**Indirect Digitization** allows users to manually input their clothes. By selecting a major category (as seen on the left screen), users can specify color and length in a new window. This method solves the issue where building a DB solely through direct methods might take too long, making it difficult for users to enjoy the service immediately.

**Direct Digitization** (shown on the right screen) builds the DB by extracting clothing items from OOTD (Outfit Of The Day) images uploaded by the user. For this, we fine-tuned **YOLO** [[README]](model/YOLO/README.md) and **EfficientNet** [[README]](model/EfficientNet/README.md) models using the **K-fashion dataset** [[Link]](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=51). When a user uploads an image, the YOLO model detects and crops the bounding boxes for each clothing item, and the EfficientNet model extracts the specific attributes of the clothes within those boxes.

## Limitations and Future Work


### 1. Closet DB Construction via Order History
We planned a feature where users could upload screenshots of their order history from fashion platforms to be digitized. Although we successfully extracted text, we struggled to accurately identify specific clothing items based on that text. This remains a task for future development.

### 2. EfficientNet Training Optimization
The K-fashion dataset used for training EfficientNet is massive, containing over 1 million images. Handling such a large dataset presented technical challenges, and due to time constraints after troubleshooting, the model was not fully optimized. 

### 3. Purchase Recommendations
The goal is to recommend which clothes to buy based on the stored style preferences and existing closet DB. Due to time constraints, we were unable to link this task to the live database. This will be implemented in the future.

## Team Composition

| Name | Track | Role |
| :--- | :--- | :--- |
| **Seoyoung Choi (Lead)** | DS 25 | Product Manager, YOLO Fine-tuning, K-fashion Preprocessing, Data Crawling |
| **Jungyang Park** | DS 24 | Embedding Model Training, YOLO Fine-tuning, DB Integration, Data Crawling |
| **Jaebin Jeong** | DS 25 | K-fashion Preprocessing, EfficientNet Training |
| **Minseo Kim** | DA 25 | Front-End, Data Crawling, UX/UI Design |
| **Dogeun Lim** | DA 25 | Front-End (Direct DB), Data Crawling |
| **Isaac Jung** | DE 25 | Back-End, Data Crawling |
