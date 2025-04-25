# Product Price Forecasting System

## Overview
The Product Price Forecasting System is an advanced machine learning solution for predicting product prices based on various features such as ratings, reviews, categories, and textual descriptions. This system combines multiple regression models, feature engineering, and natural language processing to deliver accurate price predictions and market insights.

## Problem Statement
In today's dynamic e-commerce landscape, accurately predicting product prices is crucial for:

- Sellers looking to optimize their pricing strategy
- Buyers wanting to determine if they're getting a fair deal
- Marketplace analysts tracking price trends across categories
- Inventory managers making purchasing decisions

Traditional pricing methods often rely on limited data points and fail to capture the nuances of how features like product descriptions, ratings, and review sentiment affect pricing. This system addresses these challenges by using machine learning to identify complex patterns across multiple data dimensions.

## Features

### Multi-model Approach
- Combines Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, and Ridge regression models
- Stacked Ensemble: Uses stacking to improve prediction accuracy by leveraging strengths of individual models
- Automated Feature Engineering: Extracts valuable information from text data using sentiment analysis

### Interactive Interfaces
- Text-to-text chat for querying insights
- Speech-to-speech interface for hands-free operation
- Manual data entry or batch file processing

### Additional Capabilities
- Data Flexibility: Supports various input formats including CSV, JSON, JSONL, and Parquet files
- Visualization: Generates detailed visualizations for model performance and feature importance
- Knowledge Base: Creates a searchable knowledge base for retrieving relevant product information

## System Architecture
The system follows a modular pipeline architecture:

### Data Loading & Preprocessing
- Handles missing values
- Performs feature engineering (text features, category encoding)
- Applies transformations for skewed numerical features

### Model Training
- Cross-validates multiple regression models
- Optimizes hyperparameters through grid search
- Evaluates models on test data

### Ensemble Learning
- Combines models using stacking technique
- Leverages XGBoost as a meta-model

### Knowledge Base Creation
- Vectorizes product information using TF-IDF
- Enables semantic search for similar products

### Interactive Interface
- Provides chat-based and voice-based interfaces
- Generates contextual responses using LLM integration

## Getting Started

### Prerequisites
- Python 3.11+
- Required packages (automatically installed when running the system):
  - pandas, numpy, scikit-learn
  - xgboost, lightgbm, catboost
  - textblob (for sentiment analysis)
  - matplotlib, seaborn (for visualization)
  - pyttsx3, sounddevice (for speech interface)
  - google.generativeai (for LLM integration)

### Installation

Clone the repository:
```bash
git clone https://github.com/ahmedmohamed7334/Sales_Forcast_Project
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script:
```bash
python Modeling.py
```


## Data Format
The system works with structured product data containing some of the following fields:

- `asin`: Product identifier
- `title`: Product title
- `main_category`: Product category
- `price` or `log_price`: Product price (target variable)
- `rating`: Product rating (e.g., 1-5 scale)
- `total_reviews`: Number of reviews
- `text`: Product description or review text

Not all fields are required, as the system can handle missing data.

## Example Queries
With the chatbot interface, you can ask questions like:

- "What's the average predicted price for electronics?"
- "How do features affect the price of kitchen products?"
- "Compare prices between books and electronics"
- "Tell me about products with high ratings"
- "What factors most influence product pricing?"

## Model Training Process
The system:

1. Splits data into training and test sets
2. Performs preprocessing and feature engineering
3. Trains multiple regression models with cross-validation
4. Evaluates models using MSE, RMSE, MAE, and R²
5. Creates visualizations of model performance
6. Builds a stacking ensemble for final predictions

## Speech Interface
The speech interface allows for hands-free operation:

1. Press '1' to start recording
2. Ask your question verbally
3. Press '2' to stop recording
4. The system transcribes your speech, processes the query, and responds verbally

## Development and Extension
The modular architecture makes it easy to:

- Add new ML models to the ensemble
- Extend feature engineering capabilities
- Integrate with other data sources
- Customize the visualization components
- Enhance the chatbot's response generation

## License
This project is open source and available under the MIT License.

## Acknowledgments
- Special thanks to Islam Adel for his guidance and instruction
- Thanks to DEPI for providing this opportunity to learn
- Thanks to the open-source ML community for the excellent libraries
- Special thanks to contributors to the scikit-learn, XGBoost, LightGBM, and CatBoost projects