# TRANSFER LEARNING & MODEL FINE TUNING
This project explores transfer learning for handwritten letter classification using a pretrained PyTorch MobileNetV2 model and the EMNIST Letters dataset.
The objective of the project was to adapt a convolutional neural network pretrained on ImageNet to recognize handwritten English letters (A–Z). Two transfer learning approaches were implemented and compared:
* Feature extraction using a frozen pretrained backbone
* Fine-tuning the pretrained network
## PROJECT OVERVIEW
The EMNIST Letters dataset contains grayscale handwritten character images representing the 26 English letters. Since MobileNetV2 expects RGB images at a larger input resolution, the dataset was preprocessed by:
* Resizing images to 224×224
* Converting grayscale images into 3-channel RGB images
* Applying ImageNet normalization
A custom classifier head was added to the pretrained MobileNetV2 architecture to perform 26-class letter classification.
## TRANSFER LEARNING APPROACHES 
* In the first experiment, the pretrained MobileNetV2 feature extractor was frozen so that only the newly added classifier layers were trained.
* In the first experiment, the pretrained MobileNetV2 feature extractor was frozen so that only the newly added classifier layers were trained.
## TECH PACKAGE
* Python
* PyTorch
* Torchvision
* Matplotlib
* tqdm

## RESULTS
The project compared the performance of:
* Frozen pretrained features
* Fully fine-tuned pretrained features
It was found that fine-tuning generally improved classification performance by allowing the pretrained ImageNet representations to better adapt to handwritten character recognition. The fine-tuned unfrozen features model achived a test accurary of **92.55%** compared to **88.98%** for the model with frozen features
