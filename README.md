# Final_Project
Project 1:This project compares seven popular CNN architectures, Project 2: Sequence-to-Sequence (Seq2Seq) model with an Attention mechanism, Project 3: This project is a multifunctional NLP and image generation tool built using Hugging Face models.

Project 1:
CNN Model Comparison: LeNet-5, AlexNet, VGG, ResNet, Xception, GoogLeNet, SENet
This project implements and compares multiple convolutional neural network architectures on popular image classification datasets using PyTorch.

1. Objective
To analyze and compare the performance of 7 deep learning models — LeNet-5, AlexNet, VGGNet, ResNet, GoogLeNet, Xception, and SENet — across MNIST, FashionMNIST, and CIFAR-10 datasets using key evaluation metrics.

2. What It Includes
Custom and pretrained CNN architectures

Unified training and evaluation pipeline

Metric tracking: Accuracy, Precision, Recall, F1-Score, and Loss

Visualization of model performance

Final comparison table and plots

3. Technologies Used
Python, PyTorch

TorchVision for datasets and models

Scikit-learn for metric computation

Matplotlib and pandas for visualization and tabulation

4. Results
Each model is evaluated and compared on all datasets. Results are stored in structured formats and visualized via performance plots. Best models per dataset can be easily identified.

Project2:
Seq2Seq Model with Attention Mechanism
This project implements a Sequence-to-Sequence (Seq2Seq) model enhanced with an Attention Mechanism using PyTorch. It trains on a synthetic dataset where the goal is to reverse sequences of integers, demonstrating the power of attention in improving sequence learning.

1. Project Objective
Implement and evaluate a neural seq2seq architecture with attention to show how attention helps models focus on relevant input elements during decoding.

2. Dataset
Synthetic data: sequences of random integers (e.g., [3, 1, 7])

Target output: reversed version (e.g., [7, 1, 3])

Automatically generated — no external data needed.

3. Model Architecture
Encoder: Embedding + LSTM

Attention: Additive attention mechanism

Decoder: Embedding + Attention + LSTM + Linear

Seq2Seq: Combines encoder-decoder with teacher forcing

4. Training & Evaluation
Optimizer: Adam

Loss: CrossEntropyLoss

Accuracy and loss tracked across epochs

Visualizations include training loss and accuracy plots

5. Key Outcomes
Achieved ~97% accuracy in reversing sequences

Demonstrated how attention improves alignment and learning

Easily extendable to real NLP tasks like translation or summarization

Project 3:
Multifunctional NLP and Image Generation Tool
1. All-in-One AI Interface
This project combines multiple Natural Language Processing (NLP) tasks and AI-powered image generation into a single interactive web app using Streamlit and Hugging Face models.

2. Supported Tasks
Users can perform tasks such as text summarization, next-word prediction, story generation, sentiment analysis, chatbot conversations, question answering, and text-to-image generation using Stable Diffusion.

3. Pretrained Hugging Face Models
The application leverages state-of-the-art pretrained models from the Hugging Face Transformers and Diffusers libraries to provide fast, accurate, and meaningful outputs.

5. User-Friendly Frontend
A clean and intuitive Streamlit-based interface allows users to easily switch between tasks, input text or prompts, and visualize outputs in real time—including generated images.





