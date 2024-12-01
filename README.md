# PlantDiseases_CNN_Classification
Classification of Plant Diseases Using CNN Algorithms

## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify plant diseases based on leaf images. The main goal is to develop an accurate and efficient model for identifying various plant diseases, which can be crucial for early detection and management in agriculture.

## Dataset
The project utilizes the Plant Village dataset, a comprehensive collection of plant leaf images. This dataset includes:
- Images of both healthy and diseased plant leaves
- Multiple plant species
- Various disease categories
- High-quality, labeled images suitable for machine learning tasks

## Notebook Contents (CNN_PD.ipynb)
### 1. Data Preparation
- **Data Loading**: Scripts to load and organize the Plant Village dataset
- **Data Splitting**: Techniques for dividing data into training, validation, and test sets
- **Data Augmentation**: Implementation of image augmentation techniques such as rotation, flipping, and zooming to enhance model robustness

### 2. Model Architecture
- **CNN Structure**: Detailed implementation of a custom CNN using TensorFlow and Keras
- **Layer Configuration**: 
  - Multiple convolutional layers for feature extraction
  - Max pooling layers for spatial dimension reduction
  - Dense layers for classification
- **Regularization**: Incorporation of dropout layers to prevent overfitting

### 3. Model Training
- **Compilation**: Setting up the model with appropriate loss function (likely categorical cross-entropy) and optimizer (e.g., Adam)
- **Training Process**: Iterative training of the model on the prepared dataset
- **Progress Monitoring**: Real-time tracking of training and validation accuracy/loss

### 4. Model Evaluation
- **Performance Metrics**: Comprehensive evaluation on the test set, including accuracy, precision, recall, and F1-score
- **Confusion Matrix**: Visual representation of the model's classification performance
- **Learning Curves**: Plots of training and validation accuracy/loss over epochs

### 5. Prediction and Visualization
- **Sample Predictions**: Demonstration of the model's predictions on new, unseen images
- **Visualization Tools**: Functions to display input images alongside predictions and actual labels
- **Attention Maps**: (Optional) Visualization of areas the model focuses on for making predictions

### 6. Results and Analysis
- **Performance Summary**: Detailed discussion of the model's overall performance
- **Error Analysis**: In-depth look at misclassifications and potential reasons
- **Comparative Analysis**: (If applicable) Comparison with baseline models or state-of-the-art results

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- OpenCV (cv2)
- Pandas
- Scikit-learn

## Installation and Setup
1. Clone the repository:
   ```
   git clone https://github.com/Jakee4488/PlantDiseases_CNN_Classification.git
   ```
2. Navigate to the project directory:
   ```
   cd PlantDiseases_CNN_Classification
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Ensure all dependencies are installed
2. Download the Plant Village dataset and place it in the `data` directory
3. Open and run the `CNN_PD.ipynb` notebook:
   ```
   jupyter notebook CNN_PD.ipynb
   ```
4. Follow the notebook cells sequentially for a step-by-step execution of the project

## Future Work
- Implementation of more advanced CNN architectures (e.g., ResNet, Inception)
- Exploration of transfer learning techniques using pre-trained models
- Development of strategies to handle class imbalance in the dataset
- Creation of a web-based or mobile application for real-time plant disease diagnosis
- Integration with IoT devices for automated monitoring in agricultural settings

## Contributors
- Jake E. (Project Lead)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Plant Village dataset creators and contributors
- TensorFlow and Keras development teams
- Open-source community for various tools and libraries used in this project

For any questions or collaborations, please open an issue or contact [your-email@example.com].
