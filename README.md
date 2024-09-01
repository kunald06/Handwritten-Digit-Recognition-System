# Handwritten Digit Recognition using K-Nearest Neighbors

This project aims to recognize handwritten digits using the K-Nearest Neighbors (KNN) algorithm. Handwritten digit recognition is a fundamental problem in the field of computer vision and machine learning, with applications ranging from digitizing handwritten documents to enabling automation in postal services.

## Overview

The project utilizes the MNIST dataset, a widely used benchmark dataset in the field of machine learning. The dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9), along with their corresponding labels. KNN, a simple yet effective machine learning algorithm, is employed for classification.

## Project Structure

The project consists of the following components:

1. **Data Loading and Preprocessing:** The MNIST dataset is loaded using TensorFlow's Keras API. The images are reshaped and normalized to scale pixel values between 0 and 1.

2. **Model Training:** The KNN classifier is trained on the preprocessed training data. Feature scaling is applied to standardize the feature values.

3. **Model Evaluation:** The trained model is evaluated on the test data. Performance metrics such as confusion matrix and accuracy score are calculated to assess the model's effectiveness.

4. **Visualization:** Sample images from the training and test sets are randomly selected and visualized along with their true and predicted labels.

5. **Results Analysis:** The predictions are saved to a CSV file for further analysis if needed.

## Dependencies

- Python 3.x
- TensorFlow
- NumPy
- pandas
- matplotlib
- scikit-learn

## Usage

1. Ensure all dependencies are installed. You can install them via pip:

    ```
    pip install tensorflow numpy pandas matplotlib scikit-learn
    ```

2. Open the Jupyter Notebook `Notebook.ipynb`.

3. Execute each cell in the notebook sequentially.

4. The notebook will train the KNN model, evaluate its performance on the test data, and visualize sample predictions.

## File Descriptions

- `Notebook.ipynb`: Jupyter Notebook containing the implementation of the handwritten digit recognition using KNN.
- `predictions.csv`: CSV file containing the actual and predicted labels for the test data.

## Acknowledgments

- MNIST dataset: Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-Based Learning Applied to Document Recognition."
- TensorFlow: TensorFlow is an open-source machine learning framework developed by Google.
- scikit-learn: scikit-learn is a Python library for machine learning built on NumPy, SciPy, and matplotlib.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to modify and extend this project for your own use cases! If you encounter any issues or have suggestions for improvements, please feel free to contribute.
<hr>