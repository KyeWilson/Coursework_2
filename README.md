# Galaxy Zoo Neural Network Analysis

This project explores the Galaxy Zoo dataset using machine learning and deep learning techniques. It answers three key questions:
1. **Q1**: How well can a Decision Tree classify celestial objects?
2. **Q2**: How does a Neural Network compare to a Decision Tree for this task?
3. **Q3**: How does the choice of activation function affect Neural Network performance?

The project is structured to guide beginners through data pre-processing, model design, and performance evaluation.

## Dataset
The dataset is derived from the **Galaxy Zoo Challenge** and contains information about celestial objects with features such as:
- **Features**: RA, DEC, magnitudes (u, g, r, i, z), and redshift.
- **Target**: Object classification (Galaxy, Star, or Quasar).

The dataset is cleaned and pre-processed in each notebook for reproducibility.

## Notebooks

### Q1: Decision Tree Classifier
- **Objective**: Use a Decision Tree Classifier to classify celestial objects.
- **Key Results**:
  - Accuracy: **99%**
  - High precision, recall, and F1-scores across all classes.

### Q2: Neural Network Classifier
- **Objective**: Implement a Neural Network to classify celestial objects and compare it to the Decision Tree.
- **Key Results**:
  - Accuracy: **98%**
  - Improved performance for Quasars compared to the Decision Tree.

### Q3: Effect of Activation Functions
- **Objective**: Investigate the impact of different activation functions (ReLU, Sigmoid, Tanh) on Neural Network performance.
- **Key Results**:
  - **ReLU** and **Tanh** achieved 99% accuracy.
  - **Tanh** minimised loss better (0.0505) compared to ReLU (0.0621).
  - **Sigmoid** underperformed with 97% accuracy and higher loss.

## How to Run the Notebooks

1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   pip install -r dependencies.txt
   jupyter notebook

Navigate to the respective Q1, Q2, and Q3 folders to explore the notebooks.

### Acknowledgments
- Galaxy Zoo Challenge for the dataset.
- TensorFlow and Scikit-learn for the machine learning tools.
- My coursework for inspiring this analysis.
