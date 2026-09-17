# Image Classification with Logistic Regression 🖼️

This project started with a simple idea: can a machine learning model learn to distinguish between two groups of images?

To test that idea, I created a small image dataset and used Python to turn each image into numerical data. Every image is resized to `64 × 64` pixels, converted to RGB, and then one color channel is extracted and flattened into a one-dimensional array.

The pixel values are normalized to a range between 0 and 1. These numbers become the features that are given to a Logistic Regression model.

The images are split into two classes, and the model is trained using the first set of images. After training, it can make predictions on test images and also on a completely new image that was not part of the original training set.

One of the most interesting parts of this project was seeing how an image, which looks like a picture to us, can be represented as thousands of numbers that a machine learning model can work with.

### From Image to Prediction

```text
Image
  ↓
Resize to 64 × 64
  ↓
Convert to RGB
  ↓
Extract Pixel Values
  ↓
Normalize
  ↓
Flatten into Features
  ↓
Logistic Regression
  ↓
Prediction
```

### What the Project Uses

* Python
* NumPy
* Matplotlib
* Pillow
* Scikit-learn

### Inside the Dataset

The project uses ten training images divided between two classes.

Each image is converted into a numerical feature vector containing its pixel information. The labels are then used to teach the Logistic Regression model which class each image belongs to.

After the model is trained, image `11.png` is processed in exactly the same way and passed to the model for prediction.

### A Small Example

The program also displays some of the images during the process. This makes it easier to see the connection between the original image and the numerical data being used by the model.

### Why I Built This

I wanted to understand what actually happens when machine learning is applied to an image.

Instead of using a complicated deep learning model, I started with Logistic Regression and a very small dataset. This helped me focus on the basics: reading images, extracting pixels, reshaping arrays, normalizing values, training a classifier, and making predictions.

### Running the Project

Install the required packages:

```bash
pip install numpy matplotlib pillow scikit-learn
```

Then run:

```bash
python image_classification.py
```

Make sure the image dataset exists in the path used by the program.

### What I Learned

This project gave me practical experience with image arrays, pixel values, reshaping data, normalization, train/test splitting, and classification with Logistic Regression.

It also helped me understand an important idea in machine learning: before a model can learn from an image, the image has to be represented in a form that the model can understand.
