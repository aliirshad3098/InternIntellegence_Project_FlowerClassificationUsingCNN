# Flower Classification Project

This project focuses on building a Convolutional Neural Network (CNN) model to classify images of flowers into five distinct categories. The model was trained using a dataset of images (with each category containing between 700 and 1000 images) and uses advanced deep learning techniques to enhance performance.

## Key Features

- **Data Augmentation**: To improve the model's robustness, multiple augmented samples of each image were created, allowing for better generalization.
- **CNN Architecture**: The model includes multiple convolutional layers, dropout layers to prevent overfitting, and dense layers to capture complex patterns in the data.
- **MaxPooling**: Used to reduce the spatial dimensions of the images, making the model more efficient without losing significant information.
- **Training & Validation**: Achieved 91.45% training accuracy and 89.80% validation accuracy after 30 epochs.
- Training Details
The model was trained for 30 epochs, achieving significant improvements in both training and validation accuracy:

Training Accuracy: 30.82% to 91.45%
Validation Accuracy: 46.58% to 89.80%
Loss: Consistently reduced throughout the training process.
Example Training Log
Epoch 1/30: Training Accuracy: 30.82% | Validation Accuracy: 46.58% | Validation Loss: 1.1972
Epoch 30/30: Training Accuracy: 91.45% | Validation Accuracy: 89.80% | Validation Loss: 0.3411

- **Model Deployment**: A function has been implemented for classifying new flower images based on the trained model.

## Dataset

The dataset used for this project contains five categories of flowers. Due to size constraints, the dataset has not been uploaded here. You can access it using the following link:

[Flower Dataset Link]()

## How to Run

1. Download the dataset from the link provided above.
2. Ensure that the dataset is organized into the appropriate folder structure for training and validation.
3. Train the model by running the provided Python script.

## Future Improvements

- Fine-tuning the CNN architecture for better accuracy.
- Exploring additional augmentation techniques or transfer learning to further improve performance.

## Conclusion

This project demonstrates the application of deep learning techniques in image classification and shows the potential of CNNs for handling complex visual data.

---
