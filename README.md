# Food Recognition and Calorie Estimation Model

## Project Overview
This project aims to develop a deep learning model capable of recognizing food items from images and estimating their calorie content. By leveraging this model, users can track their dietary intake more accurately, facilitating informed food choices for maintaining a healthy lifestyle.

## Dataset
The dataset used for training and testing consists of images of various food items categorized into different classes. Each class represents a specific food item, and the dataset is structured in directories where each directory corresponds to a class. The dataset is divided into training and testing sets to train and evaluate the model's performance, respectively.

## Implementation
The implementation involves building a CNN-based deep learning model for food recognition and calorie estimation from images. The dataset is split into training and testing sets, categorized into food classes. Data augmentation enhances model generalization. The model architecture includes convolutional, pooling, flatten, dropout, and dense layers. Training utilizes Adam optimizer and categorical cross-entropy loss over epochs. Evaluation metrics like accuracy, loss, confusion matrix, and classification report gauge model performance. A calorie content dictionary estimates the calorie content of recognized food items. Overall, the implementation aims to deliver a reliable solution for informed dietary choices.

## Explanation of the Code

### Import Libraries:
The code begins by importing necessary libraries for various tasks in the machine learning pipeline. These include:
- NumPy
- TensorFlow
- TensorFlow.keras
- Matplotlib
- Seaborn
- PIL

### Define Directories:
Two directory paths are defined: `train_dir` and `test_dir`, which represent the locations of the training and testing datasets, respectively. These paths are used later for loading the dataset images.

### Define Image Dimensions and Batch Size:
The variables `img_width` and `img_height` store the desired dimensions to which the input images will be resized for model training. Additionally, `batch_size` specifies the number of images to be processed in each training batch.

### Calculate Number of Classes:
The number of classes (`num_classes`) is determined by counting the subdirectories within the training directory (`train_dir`). Each subdirectory corresponds to a distinct food category, and thus represents a class.

### Data Augmentation for Training:
Data augmentation is a crucial step to increase the diversity of training data and improve model generalization. The `train_datagen` object is created using `ImageDataGenerator` from `TensorFlow.keras.preprocessing.image` module. This object applies various transformations such as rotation, shifting, shearing, zooming, and flipping to augment the training images.

### Data Preprocessing for Validation:
Similar to data augmentation, validation data needs to be preprocessed for consistency. The `test_datagen` object is created to rescale pixel values of validation images by a factor of 1/255, ensuring they lie in the range [0, 1].

### Load and Augment Training Images:
The `train_generator` is created using `flow_from_directory` method of `train_datagen`. This generator loads images from the training directory (`train_dir`), resizes them to the specified dimensions, applies data augmentation transformations, and yields batches of augmented images along with their corresponding labels during model training.

### Load Validation Images:
Similarly, the `validation_generator` is created using `flow_from_directory` method of `test_datagen`. This generator loads and preprocesses validation images from the testing directory (`test_dir`) for model evaluation during training.

### Model Architecture Definition and Compilation:
The model architecture is defined using TensorFlow.keras's Sequential API, comprising convolutional and pooling layers followed by dense layers. After defining the model, it is compiled with appropriate optimizer, loss function, and evaluation metric for training.

### Model Training:
The model is trained using the `fit` method, which iterates over batches of training data (`train_generator`) for a specified number of epochs. The training progress is monitored, and validation data (`validation_generator`) is used to evaluate the model's performance on unseen data.

### Visualization and Evaluation:
After training, the code visualizes the training and validation accuracy/loss using Matplotlib. Furthermore, it generates predictions on the test set (`test_generator`) and computes evaluation metrics such as confusion matrix and classification report to assess the model's performance.

### Calorie Content Estimation:
Lastly, the code randomly selects images from the test set, predicts their classes, and estimates their calorie content using a predefined dictionary mapping food classes to calorie values. The predicted class and calorie content are displayed for each image.
## Result

The developed model demonstrates promising performance in accurately recognizing food items from images and estimating their calorie content. The training and validation accuracy and loss plots indicate effective learning and generalization capabilities of the model. Evaluation metrics such as confusion matrix and classification report further validate the model's accuracy and robustness.

## Acknowledgement
The above dataset is from Kaggle.
