# 📰 Fake News Prediction Using Machine Learning

A machine learning pipeline for detecting fake news articles using Natural Language Processing (NLP) techniques and multiple classification algorithms. This project demonstrates comparative analysis of different ML models to accurately distinguish between real and fake news.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 📌 Project Overview

- **🎯 Objective:** Predict whether a news article is **Real (1)** or **Fake (0)** based on its title
- **📊 Dataset:** [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- **🔧 Approach:** Comparative analysis of four machine learning models with comprehensive text preprocessing
- **📈 Best Model:** Logistic Regression with **96.59%** test accuracy
- **💻 Environment:** Originally developed in Google Colab

---

## 🏗️ Project Architecture

```
Data Input → Text Preprocessing → Feature Extraction → Model Training → Evaluation → Prediction
     ↓              ↓                    ↓               ↓            ↓           ↓
WELFake Dataset → Cleaning & Stemming → TF-IDF → 4 ML Models → Accuracy → Real/Fake
```

---

## 📁 File Structure

```
fake_news_prediction/
│
├── fake_news_prediction.ipynb   # Main Jupyter Notebook
├── WELFake_Dataset.csv         # Input dataset
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies (optional)
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- Jupyter Notebook or Google Colab
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/Deepanshu-kesharwani/fake_news_prediction.git
cd fake_news_prediction
```

### Step 2: Install Dependencies
```bash
pip install numpy pandas scikit-learn nltk jupyter
```

### Step 3: Download NLTK Data
Run this in a notebook cell or Python script:
```python
import nltk
nltk.download('stopwords')
```

### Step 4: Prepare Dataset
- Download `WELFake_Dataset.csv` and place it in the project directory
- For Google Colab: Upload the dataset to `/content/` directory
- For local Jupyter: Ensure the file path matches the one in the notebook

---

## 🚀 Usage

### Option 1: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `fake_news_prediction.ipynb`
3. Upload `WELFake_Dataset.csv` to the Colab environment
4. Run all cells sequentially

### Option 2: Local Jupyter Notebook
```bash
# Start Jupyter Notebook
jupyter notebook

# Open fake_news_prediction.ipynb in your browser
# Run all cells or execute step by step
```

### Option 3: JupyterLab
```bash
# Start JupyterLab
jupyter lab

# Open and run the notebook
```

---

## 📊 Notebook Structure

The notebook is organized into the following sections:

1. **📚 Import Libraries**
   - NumPy, Pandas for data manipulation
   - NLTK for text processing
   - Scikit-learn for ML models

2. **📂 Data Loading & Exploration**
   - Load WELFake dataset
   - Handle missing values
   - Basic data exploration

3. **🧹 Data Preprocessing**
   - Text cleaning (remove non-alphabetic characters)
   - Lowercase conversion
   - Stopword removal
   - Porter stemming

4. **🔢 Feature Engineering**
   - TF-IDF vectorization
   - Train-test split (80-20)

5. **🤖 Model Training**
   - Logistic Regression
   - Decision Tree Classifier
   - Gradient Boosting Classifier
   - Random Forest Classifier

6. **📊 Model Evaluation**
   - Accuracy calculation
   - Performance comparison
   - Sample predictions

---

## 🧪 Expected Output

When you run the notebook, you'll see:

```
--- Training Accuracies ---
Logistic Regression Training Accuracy: 98.64%
Decision Tree Training Accuracy: 100.00%
Gradient Boosting Training Accuracy: 98.98%
Random Forest Training Accuracy: 100.00%

--- Test Accuracies ---
Logistic Regression Test Accuracy: 96.59%
Decision Tree Test Accuracy: 92.95%
Gradient Boosting Test Accuracy: 94.94%
Random Forest Test Accuracy: 95.98%

--- Sample Prediction ---
Prediction: [1]
The news is Real
Actual Label: 1
```

---

## 🧪 Model Performance Analysis

| Model | Training Accuracy | Test Accuracy | Overfitting Risk |
|-------|------------------|---------------|------------------|
| **Logistic Regression** | **98.64%** | **96.59%** | ✅ Low |
| Decision Tree | 100.00% | 92.95% | ⚠️ High |
| Gradient Boosting | 98.98% | 94.94% | ✅ Low |
| Random Forest | 100.00% | 95.98% | ⚠️ Medium |

### 🏆 Best Model: Logistic Regression
- **Reason:** Best balance between training and test accuracy
- **Advantage:** Minimal overfitting with strong generalization
- **Performance:** 96.59% test accuracy

---

## 🔍 Technical Implementation

### 1. Data Preprocessing Pipeline
```python
# Text Cleaning Steps:
1. Remove non-alphabetic characters
2. Convert to lowercase
3. Remove stopwords (NLTK)
4. Apply Porter Stemming
5. Handle missing values with empty strings
```

### 2. Feature Engineering
- **Technique:** TF-IDF Vectorization
- **Input:** News article titles (content column)
- **Output:** Numerical feature vectors
- **Advantage:** Captures word importance across documents

### 3. Model Training
```python
# Models Implemented:
- LogisticRegression()
- DecisionTreeClassifier()
- GradientBoostingClassifier()
- RandomForestClassifier()
```

### 4. Evaluation Metrics
- **Primary:** Accuracy Score
- **Split:** 80% Training, 20% Testing
- **Validation:** Stratified split for balanced classes
- **Random State:** 2 (for reproducibility)

---

## 📊 Dataset Information

- **Source:** WELFake Dataset
- **Features Used:** Title column only (processed as 'content')
- **Target Variable:** Label (0 = Fake, 1 = Real)
- **Preprocessing:** Missing values filled with empty strings
- **File Location:** `/content/WELFake_Dataset.csv` (Google Colab)

---

## 🔮 Future Enhancements

### 📈 Model Improvements
- [ ] Implement deep learning models (LSTM, BERT, GPT)
- [ ] Add cross-validation for robust evaluation
- [ ] Include precision, recall, F1-score metrics
- [ ] Implement ensemble methods
- [ ] Add confusion matrix visualization

### 🔧 Feature Engineering
- [ ] Use full article content instead of just titles
- [ ] Add author information and publication source
- [ ] Implement advanced NLP techniques (NER, POS tagging)
- [ ] Apply word embeddings (Word2Vec, GloVe)

### 📱 Deployment Options
- [ ] Convert to Streamlit web app
- [ ] Build REST API with Flask/FastAPI
- [ ] Deploy on cloud platforms (AWS, Heroku)
- [ ] Create interactive dashboard

### 📊 Visualization Enhancements
- [ ] Add data exploration plots
- [ ] Model performance comparison charts
- [ ] Word cloud visualizations
- [ ] Feature importance plots

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes in the notebook
4. Test thoroughly
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### 📋 Contribution Guidelines
- Follow PEP 8 style guide in code cells
- Add markdown documentation for new sections
- Include comments in code cells
- Test all cells before submitting

---

## 🐛 Troubleshooting

### Common Issues:

1. **Dataset not found error:**
   - Ensure `WELFake_Dataset.csv` is in the correct directory
   - For Colab: Upload to `/content/` directory
   - For local: Adjust path in the notebook

2. **NLTK stopwords error:**
   - Run `nltk.download('stopwords')` in a code cell

3. **Memory issues:**
   - Use Google Colab Pro for more RAM
   - Or reduce dataset size for testing

4. **Import errors:**
   - Install required packages: `!pip install package_name` in Colab
   - Or use `pip install` in local environment

---

## 📚 References & Resources

- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Google Colab Documentation](https://colab.research.google.com/)
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Deepanshu Kesharwani**
- GitHub: [@Deepanshu-kesharwani](https://github.com/Deepanshu-kesharwani)
- LinkedIn: [Connect with me](https://linkedin.com/in/deepanshu-kesharwani)

---

## 🙏 Acknowledgments

- Thanks to the creators of the WELFake dataset
- Google Colab for providing free GPU/TPU access
- NLTK and Scikit-learn communities for excellent documentation
- Open source contributors who made this project possible

---

<div align="center">
  <p>⭐ Star this repository if you found it helpful!</p>
  <p>🐛 Found a bug? Open an issue!</p>
  <p>💡 Have suggestions? Let's discuss!</p>
</div>
