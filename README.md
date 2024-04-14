# Lungcancer_prediction_OneDnn
![04-Blog-Tumor-Cell-S](https://user-images.githubusercontent.com/111365771/222963183-1b677b56-822a-4b05-8b73-3d48a0c13db3.jpg)

# Lung Cancer Classification Project!
This project aims to classify lung cancer types based on histopathological images. The goal is to develop a deep learning model that can accurately predict the type of lung cancer present in a given image.


### Introduction

Lung cancer remains one of the leading causes of cancer-related deaths worldwide, with various subtypes presenting unique challenges in diagnosis and treatment. Histopathological examination of lung tissue samples plays a crucial role in diagnosing and classifying different types of lung cancer. However, manual interpretation of histopathological images by pathologists can be time-consuming and subjective, leading to variability in diagnoses.

To address these challenges, this project focuses on developing a deep learning model for the automated classification of lung cancer types based on histopathological images. By leveraging the power of convolutional neural networks (CNNs), we aim to create a robust and accurate system capable of identifying adenocarcinoma, squamous cell carcinoma, small cell carcinoma, and other lung cancer subtypes with high precision.


### Explanation

The project begins by curating a dataset of histopathological images of lung cancer samples, carefully annotated with their corresponding cancer types. This dataset is divided into training, validation, and test sets to facilitate model development and evaluation.
![fcae756d-ca48-4c09-a594-596728cee8f6](https://github.com/Vijay18003/Lungcancer_prediction_OneDnn/assets/158248736/bf8d0992-d642-45bd-9d22-2b6a065c81c3)

For the model architecture, we employ the EfficientNetB2 architecture as a base model, pre-trained on the ImageNet dataset. This architecture offers a good balance between computational efficiency and model performance. We fine-tune the pre-trained model on our lung cancer dataset by adding additional layers for classification.
![57bfc84c-eb4d-4f52-8f78-8a1eb1f7189c](https://github.com/Vijay18003/Lungcancer_prediction_OneDnn/assets/158248736/15bf6ce5-8a60-4deb-a7db-d779b87cd65f)

During the training process, we utilize the Adamax optimizer with a categorical cross-entropy loss function. Data augmentation techniques such as horizontal flipping are applied to increase the diversity of training samples and improve the model's generalization ability. Additionally, a custom learning rate scheduler is implemented to adjust the learning rate based on training and validation metrics, optimizing the model's performance.

After training, the model is evaluated on a separate test set to assess its ability to accurately classify lung cancer types. Evaluation metrics such as accuracy, precision, recall, and F1-score are calculated to measure the model's performance. Furthermore, a confusion matrix is generated to visualize the model's performance across different classes, providing insights into its strengths and weaknesses.



## Dataset

The dataset used in this project consists of histopathological images of lung cancer samples. It contains images from multiple classes, including adenocarcinoma, squamous cell carcinoma, small cell carcinoma, and others. The dataset is divided into training, validation, and test sets.

## Model Architecture

The model architecture used for this project is based on the EfficientNetB2 architecture pre-trained on the ImageNet dataset. The pre-trained base model is fine-tuned on the lung cancer dataset by adding additional layers for classification. The final layer uses softmax activation to output probabilities for each class.
![197c6432-3811-4b0d-aa93-66c736543c92](https://github.com/Vijay18003/Lungcancer_prediction_OneDnn/assets/158248736/17b268bf-c5a9-4cc0-834d-4710a6aa131c)

## Training

The model is trained using the Adamax optimizer with a categorical cross-entropy loss function. The training process includes data augmentation techniques such as horizontal flipping to increase the diversity of training samples. Learning rate adjustment is implemented using a custom learning rate scheduler based on training and validation metrics.
![7f780ab4-12b0-40ef-8817-3de76673191a](https://github.com/Vijay18003/Lungcancer_prediction_OneDnn/assets/158248736/d364f4ec-78a2-40ea-a6e1-62a38949b67b)

## Evaluation

The trained model is evaluated on a separate test set to assess its performance. Evaluation metrics include accuracy, precision, recall, and F1-score. Additionally, a confusion matrix is generated to visualize the model's performance across different classes.

## Results

The model achieves an accuracy of 90% on the test set, demonstrating its effectiveness in classifying lung cancer types. The classification report provides detailed metrics for each class, highlighting the model's strengths and weaknesses.

## Usage

To use the trained model for inference on new images, follow these steps:

1. *Preprocessing*: Preprocess the input image to match the required input size and format.
2. *Model Loading*: Load the trained model using TensorFlow or Keras.
3. *Inference*: Feed the preprocessed image into the model and obtain the predicted class probabilities.
4. *Post-processing*: Optionally, post-process the output probabilities (e.g., thresholding) to obtain the final predicted class.

#OneAPI oneDNN:

![download](https://user-images.githubusercontent.com/111365771/222963211-f7f2d17c-14d2-49e4-b4fe-0fa2394af262.jpg)

OneAPI oneDNN is a deep learning library designed for high-performance machine learning applicati

### Conclusion

In conclusion, this project demonstrates the feasibility and effectiveness of using deep learning techniques for automated lung cancer classification based on histopathological images. The developed model achieves promising results in accurately identifying different lung cancer subtypes, showcasing its potential as a valuable tool in clinical practice.

Moving forward, further refinement and optimization of the model could lead to even higher performance and reliability. Additionally, integration of the model into existing healthcare systems could streamline the diagnostic process, providing clinicians with timely and accurate insights to guide patient care decisions.

Overall, this project represents a significant step towards leveraging artificial intelligence in the fight against lung cancer, ultimately contributing to improved patient outcomes and quality of life.

---
