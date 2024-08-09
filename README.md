
# CIFAR-10 Convolutional Neural Network (CNN)

This repository contains a Python implementation of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. The model is built using TensorFlow and Keras.

## Overview

The CIFAR-10 dataset is a widely-used benchmark in the field of machine learning and computer vision. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal of this project is to build and train a CNN that can accurately classify images into one of these 10 categories.

## Dataset

The CIFAR-10 dataset can be downloaded from [here](https://www.cs.toronto.edu/~kriz/cifar.html).

The dataset contains the following files:
- `data_batch_1` to `data_batch_5`: Training data (50,000 images)
- `test_batch`: Test data (10,000 images)
- `batches.meta`: Metadata about the dataset

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install tensorflow keras numpy matplotlib
```

## Model Architecture

The CNN model is composed of the following layers:

- Convolutional Layer: 32 filters, 3x3 kernel, ReLU activation
- Max Pooling Layer: 2x2 pool size
- Convolutional Layer: 64 filters, 3x3 kernel, ReLU activation
- Max Pooling Layer: 2x2 pool size
- Convolutional Layer: 64 filters, 3x3 kernel, ReLU activation
- Flatten Layer
- Dense Layer: 64 units, ReLU activation
- Dense Layer: 10 units, Softmax activation (output layer)

## Training

The model is trained for 10 epochs with a batch size of 32. The loss function used is `sparse_categorical_crossentropy`, and the optimizer is `adam`.

```python
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
```

## Evaluation

After training, the model is evaluated on the test data to determine its accuracy:

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")
```

## Results

The model achieves a test accuracy of approximately 70% after 10 epochs. Further improvements can be made by tuning hyperparameters, adding regularization, and using data augmentation.

## Visualization

The training process can be visualized by plotting the accuracy and loss over epochs:

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
```

## Usage

Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/cifar10-cnn.git
cd cifar10-cnn
```

Run the Python script to train the model:

```bash
python cifar10_cnn.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The CIFAR-10 dataset was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

## Citation

If you use this code, please cite the CIFAR-10 dataset as follows:

```
@TECHREPORT{Krizhevsky2009LearningML,
    author = {Alex Krizhevsky and Vinod Nair and Geoffrey Hinton},
    title = {Learning Multiple Layers of Features from Tiny Images},
    institution = {},
    year = {2009}
}
```

