# Advanced Facial Recognition System Using Deep Learning

## Project Overview
This project aims to create a high-performance facial recognition system using convolutional neural networks (CNNs) with PyTorch. It focuses on preprocessing, augmenting facial images, and training a CNN model to recognize faces with high accuracy.

## Features
- **Data Preprocessing and Augmentation:** Custom dataset handling, image transformations for augmentation including flipping and sharpening, to improve model robustness.
- **CNN Architecture:** A deep learning model with convolutional layers, batch normalization, ReLU activations, pooling, and fully connected layers for facial recognition.
- **Model Training and Evaluation:** Detailed training process with visual progress, evaluation of the model using accuracy and F1 score, and insights into model performance on the YaleB dataset.

## Technologies Used
- Python
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- NumPy
- matplotlib
- tqdm

## How to Run This Project
1. **Setup Environment:** Ensure Python and PyTorch are installed. Clone this repository to your local machine.
2. **Prepare the Dataset:** Download the YaleB dataset and organize the images in the specified directory structure.
3. **Train the Model:** Run the training script to start the training process. Adjust the hyperparameters as needed for better performance.
4. **Evaluate the Model:** After training, run the evaluation script to see the model's accuracy and F1 score on the test set.

## Project Structure
- `dataset.py`: Custom Dataset class for loading and preprocessing images.
- `model.py`: Definition of the CNN architecture for facial recognition.
- `train.py`: Script for training the model.
- `evaluate.py`: Script for evaluating the trained model on the test dataset.

## Results
Our CNN model achieved an accuracy of XX% and an F1 score of XX% on the YaleB test dataset, demonstrating its effectiveness in facial recognition tasks.

## Future Work
- Explore more advanced CNN architectures and training techniques.
- Implement real-time facial recognition.
- Test the model on more diverse and larger datasets for improved robustness and accuracy.

## Acknowledgments
Thanks to the creators of the YaleB dataset and the PyTorch community for their invaluable resources.

## License
This project is open-source and available under the MIT License.
