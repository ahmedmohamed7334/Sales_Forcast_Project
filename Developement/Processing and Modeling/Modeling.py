# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import os
import joblib

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor

# For RAG system
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Libraries imported successfully!")

# Create directories for saving models and data
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 2. Data Loading and Initial Exploration
def load_data(file_path):
    """Load and perform initial data exploration"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print("\nColumn types:")
    print(df.dtypes)
    
    print("\nSample data:")
    print(df.head())
    
    print("\nMissing values per column:")
    print(df.isna().sum())
    
    return df

# Load the dataset
df = load_data('data_set.csv')  # Update with your actual path if needed

# 3. Data Preprocessing
def preprocess_data(df, target_column='log_price', test_size=0.2, random_state=42):
    """
    Preprocess the data for machine learning
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataset
    target_column : str
        The column to predict
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test, preprocessor
    """
    print("Starting data preprocessing...")
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    # Extract features from text using text length
    if 'text' in df.columns and 'text_length' not in df.columns:
        df['text_length'] = df['text'].fillna('').apply(len)
    
    # Define features for the model
    categorical_features = [col for col in ['domain', 'main_category', 'sentiment_category'] 
                           if col in df.columns]
    
    numerical_features = [col for col in ['rating', 'sentiment_score', 'year', 'month', 'day', 
                                         'dayofweek', 'text_length'] 
                         if col in df.columns]
    
    print(f"Using categorical features: {categorical_features}")
    print(f"Using numerical features: {numerical_features}")
    
    # Create preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split data into features and target
    X = df[numerical_features + categorical_features]
    y = df[target_column]
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features

# Run preprocessing
X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features = preprocess_data(df)

# 4. Define and Train Base Models
def train_base_models(X_train, y_train, preprocessor, cv=5):
    """
    Train multiple base models with cross-validation
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Training features
    y_train : pandas Series
        Target variable
    preprocessor : ColumnTransformer
        Feature preprocessing pipeline
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    Dictionary of trained model pipelines
    """
    print("\nTraining base models with cross-validation...")
    
    # Define base models
    base_models = {
        'random_forest': RandomForestRegressor(random_state=42),
        'gradient_boosting': GradientBoostingRegressor(random_state=42),
        'ridge': Ridge(random_state=42),
        'svr': SVR()
    }
    
    # Parameters for grid search
    param_grids = {
        'random_forest': {
            'model__n_estimators': [50, 100],
            'model__max_depth': [None, 10, 20]
        },
        'gradient_boosting': {
            'model__n_estimators': [50, 100],
            'model__learning_rate': [0.01, 0.1]
        },
        'ridge': {
            'model__alpha': [0.1, 1.0, 10.0]
        },
        'svr': {
            'model__C': [0.1, 1.0, 10.0],
            'model__kernel': ['linear', 'rbf']
        }
    }
    
    # Train with grid search for each model
    trained_models = {}
    cv_results = {}
    
    for name, model in base_models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline with preprocessor and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Use grid search to find best parameters
        grid_search = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Store the best model
        trained_models[name] = grid_search.best_estimator_
        cv_results[name] = {
            'best_score': -grid_search.best_score_,  # Convert back to MSE
            'best_params': grid_search.best_params_
        }
        
        print(f"Best {name} score (MSE): {-grid_search.best_score_:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
    
    return trained_models, cv_results

# Train base models
trained_models, cv_results = train_base_models(X_train, y_train, preprocessor)

# 5. Evaluate Base Models
def evaluate_models(models, X_test, y_test):
    """
    Evaluate models on test data
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained model pipelines
    X_test : pandas DataFrame
        Test features
    y_test : pandas Series
        Test target variable
        
    Returns:
    --------
    Dictionary of evaluation metrics
    """
    print("\nEvaluating models on test data...")
    
    evaluation = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        evaluation[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{name} - Actual vs Predicted')
        plt.show()
    
    return evaluation

# Evaluate base models
evaluation = evaluate_models(trained_models, X_test, y_test)

# 6. Build Stacking Model (Supervisor)
def build_stacking_model(base_models, X_train, y_train, X_test, y_test):
    """
    Build a stacking model that combines base models
    
    Parameters:
    -----------
    base_models : dict
        Dictionary of trained base models
    X_train, y_train : Training data
    X_test, y_test : Test data
    
    Returns:
    --------
    Trained stacking model and its evaluation
    """
    print("\nBuilding stacking model (supervisor)...")
    
    # Prepare base estimators for stacking
    estimators = [(name, model) for name, model in base_models.items()]
    
    # Create stacking regressor
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(random_state=42),
        cv=5,
        n_jobs=-1
    )
    
    # Train stacking model
    stacking_regressor.fit(X_train, y_train)
    
    # Evaluate stacking model
    y_pred = stacking_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    stacking_evaluation = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print("\nStacking Model Evaluation:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    
    # Plot actual vs predicted for stacking model
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Stacking Model - Actual vs Predicted')
    plt.show()
    
    # Compare all models
    all_evaluations = evaluation.copy()
    all_evaluations['stacking'] = stacking_evaluation
    
    metrics = ['RMSE', 'MAE', 'R2']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        model_names = list(all_evaluations.keys())
        metric_values = [all_evaluations[model][metric] for model in model_names]
        
        # Create bar plot
        sns.barplot(x=model_names, y=metric_values)
        plt.title(f'Comparison of Models - {metric}')
        plt.ylabel(metric)
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    return stacking_regressor, stacking_evaluation

# Build stacking model
stacking_model, stacking_evaluation = build_stacking_model(trained_models, X_train, y_train, X_test, y_test)

# 7. Save Models and Preprocessing Pipeline
def save_pipeline(base_models, stacking_model, preprocessor, 
                 numerical_features, categorical_features, 
                 cv_results, evaluation, stacking_evaluation):
    """
    Save all components of the machine learning pipeline
    
    Parameters:
    -----------
    Various model components and metadata
    
    Returns:
    --------
    Dictionary of saved file paths
    """
    print("\nSaving machine learning pipeline components...")
    
    saved_files = {}
    
    # Save base models
    for name, model in base_models.items():
        model_path = f'models/{name}_model.pkl'
        joblib.dump(model, model_path)
        saved_files[f'{name}_model'] = model_path
    
    # Save stacking model
    stacking_path = 'models/stacking_model.pkl'
    joblib.dump(stacking_model, stacking_path)
    saved_files['stacking_model'] = stacking_path
    
    # Save preprocessor
    preprocessor_path = 'models/preprocessor.pkl'
    joblib.dump(preprocessor, preprocessor_path)
    saved_files['preprocessor'] = preprocessor_path
    
    # Save feature lists
    features = {
        'numerical_features': numerical_features,
        'categorical_features': categorical_features
    }
    features_path = 'models/feature_lists.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)
    saved_files['features'] = features_path
    
    # Save evaluation results
    results = {
        'cv_results': cv_results,
        'evaluation': evaluation,
        'stacking_evaluation': stacking_evaluation
    }
    results_path = 'models/evaluation_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    saved_files['results'] = results_path
    
    # Save model metadata
    metadata = {
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_names': list(base_models.keys()),
        'feature_count': len(numerical_features) + len(categorical_features),
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'target': 'log_price'
    }
    metadata_path = 'models/model_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    saved_files['metadata'] = metadata_path
    
    print("Pipeline components saved successfully!")
    return saved_files

# Save pipeline
saved_files = save_pipeline(
    trained_models, 
    stacking_model, 
    preprocessor, 
    numerical_features, 
    categorical_features, 
    cv_results, 
    evaluation, 
    stacking_evaluation
)

# 8. Load Models and Make Predictions
def load_pipeline():
    """
    Load all components of the saved machine learning pipeline
    
    Returns:
    --------
    Dictionary of loaded pipeline components
    """
    print("\nLoading machine learning pipeline components...")
    
    pipeline = {}
    
    # Load metadata to get model names
    with open('models/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    pipeline['metadata'] = metadata
    
    # Load base models
    base_models = {}
    for name in metadata['model_names']:
        model_path = f'models/{name}_model.pkl'
        base_models[name] = joblib.load(model_path)
    pipeline['base_models'] = base_models
    
    # Load stacking model
    pipeline['stacking_model'] = joblib.load('models/stacking_model.pkl')
    
    # Load preprocessor
    pipeline['preprocessor'] = joblib.load('models/preprocessor.pkl')
    
    # Load feature lists
    with open('models/feature_lists.pkl', 'rb') as f:
        features = pickle.load(f)
    pipeline['numerical_features'] = features['numerical_features']
    pipeline['categorical_features'] = features['categorical_features']
    
    # Load evaluation results
    with open('models/evaluation_results.pkl', 'rb') as f:
        results = pickle.load(f)
    pipeline['cv_results'] = results['cv_results']
    pipeline['evaluation'] = results['evaluation']
    pipeline['stacking_evaluation'] = results['stacking_evaluation']
    
    print("Pipeline components loaded successfully!")
    return pipeline

def make_predictions(df, pipeline, return_all_predictions=False):
    """
    Make predictions using the loaded pipeline
    
    Parameters:
    -----------
    df : pandas DataFrame
        New data for prediction
    pipeline : dict
        Loaded pipeline components
    return_all_predictions : bool
        Whether to return predictions from all models or just the stacking model
        
    Returns:
    --------
    DataFrame with predictions
    """
    print("\nPreparing data for prediction...")
    
    # Ensure we have all required features
    numerical_features = pipeline['numerical_features']
    categorical_features = pipeline['categorical_features']
    
    # Modified timestamp conversion in preprocess_data function
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    # Create text_length if text exists and text_length doesn't
    if 'text' in df.columns and 'text_length' not in df.columns:
        df['text_length'] = df['text'].fillna('').apply(len)
    
    # Select required features
    X = df[numerical_features + categorical_features].copy()
    
    print("Making predictions...")
    
    # Make predictions with each base model
    predictions = {}
    for name, model in pipeline['base_models'].items():
        predictions[f'{name}_prediction'] = model.predict(X)
    
    # Make predictions with stacking model
    predictions['stacking_prediction'] = pipeline['stacking_model'].predict(X)
    
    # Add predictions to the original dataframe
    results_df = df.copy()
    for name, preds in predictions.items():
        results_df[name] = preds
    
    # Convert log_price predictions back to price if requested
    if 'log_price' in results_df.columns:
        for name in predictions.keys():
            orig_price_col = name.replace('prediction', 'orig_price')
            results_df[orig_price_col] = np.exp(results_df[name])
    
    print("Predictions completed!")
    
    if return_all_predictions:
        return results_df
    else:
        # Return only the stacking model predictions
        selected_cols = list(df.columns) + ['stacking_prediction']
        if 'log_price' in results_df.columns:
            selected_cols.append('stacking_orig_price')
        return results_df[selected_cols]

# 9. Implement RAG (Retrieval-Augmented Generation) System
def create_knowledge_base(df):
    """
    Create a knowledge base from the dataset for retrieval
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataset with text content
        
    Returns:
    --------
    TF-IDF vectorizer and document vectors
    """
    print("\nCreating knowledge base from dataset...")
    
    # Prepare text content for knowledge base
    # Combine relevant columns into a single text field
    if 'text' in df.columns:
        # Create a combined text field with metadata
        df['kb_text'] = df.apply(
            lambda row: f"Product: {row.get('asin', 'Unknown')} | "
                      + f"Category: {row.get('main_category', 'Unknown')} | "
                      + f"Rating: {row.get('rating', 'Unknown')} | "
                      + f"Price: {np.exp(row['log_price']) if 'log_price' in df.columns else 'Unknown'} | "
                      + f"Sentiment: {row.get('sentiment_category', 'Unknown')} | "
                      + f"Text: {row.get('text', '')}",
            axis=1
        )
    else:
        # If no text column, create metadata-only knowledge base
        df['kb_text'] = df.apply(
            lambda row: f"Product: {row.get('asin', 'Unknown')} | "
                      + f"Category: {row.get('main_category', 'Unknown')} | "
                      + f"Rating: {row.get('rating', 'Unknown')} | "
                      + f"Price: {np.exp(row['log_price']) if 'log_price' in df.columns else 'Unknown'} | "
                      + f"Sentiment: {row.get('sentiment_category', 'Unknown')}",
            axis=1
        )
    
    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    # Create document vectors
    doc_vectors = tfidf_vectorizer.fit_transform(df['kb_text'])
    
    print(f"Knowledge base created with {doc_vectors.shape[0]} documents and {doc_vectors.shape[1]} features")
    
    return tfidf_vectorizer, doc_vectors, df['kb_text'].tolist()

def retrieve_relevant_documents(query, vectorizer, doc_vectors, documents, top_n=5):
    """
    Retrieve relevant documents based on the query
    
    Parameters:
    -----------
    query : str
        User query
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    doc_vectors : sparse matrix
        Document vectors
    documents : list
        Original documents
    top_n : int
        Number of documents to retrieve
        
    Returns:
    --------
    List of relevant documents
    """
    # Vectorize the query
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    
    # Get indices of top similar documents
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Get top documents and their similarity scores
    top_docs = [(documents[i], similarities[i]) for i in top_indices]
    
    return top_docs

# 10. Integration with Chatbot Functions
def generate_prediction_response(query, df_predictions, knowledge_base):
    """
    Generate a response based on the query and predictions
    
    Parameters:
    -----------
    query : str
        User query
    df_predictions : pandas DataFrame
        DataFrame with predictions
    knowledge_base : tuple
        Knowledge base components (vectorizer, doc_vectors, documents)
        
    Returns:
    --------
    Response text to be used by the chatbot
    """
    vectorizer, doc_vectors, documents = knowledge_base
    
    # Retrieve relevant documents
    relevant_docs = retrieve_relevant_documents(query, vectorizer, doc_vectors, documents)
    
    # Generate context from relevant documents
    context = "\n\n".join([f"Document (similarity: {score:.2f}):\n{doc}" 
                          for doc, score in relevant_docs])
    
    # Extract prediction stats
    avg_price = np.exp(df_predictions['stacking_prediction'].mean())
    min_price = np.exp(df_predictions['stacking_prediction'].min())
    max_price = np.exp(df_predictions['stacking_prediction'].max())
    
    prediction_stats = f"""
    Prediction Statistics:
    - Average predicted price: ${avg_price:.2f}
    - Minimum predicted price: ${min_price:.2f}
    - Maximum predicted price: ${max_price:.2f}
    """
    
    # Combine context and prediction stats to form the response
    response = f"""
    Based on your query: "{query}"
    
    {prediction_stats}
    
    Relevant information from the knowledge base:
    {context}
    """
    
    return response

def integrate_with_chatbot(df_predictions):
    """
    Integrate the prediction system with existing chatbot functions
    
    Parameters:
    -----------
    df_predictions : pandas DataFrame
        DataFrame with predictions
        
    Returns:
    --------
    None - this function calls the chatbot functions
    """
    print("\nIntegrating prediction system with chatbot...")
    
    # Create knowledge base
    knowledge_base = create_knowledge_base(df_predictions)
    
    # Define a wrapper function to be used by the chatbot
    def prediction_chatbot_handler(query):
        # Generate response based on the query and predictions
        response = generate_prediction_response(query, df_predictions, knowledge_base)
        return response
    
    # Set up the prediction response generator
    print("Setting up prediction response generator...")
    # This would be integrated with your existing generate_chat_response_v2 function
    
    # Start the audio chatbot
    print("\nStarting audio chatbot interaction...")
    print("Use '1' to start and '2' to stop the chatbot.")
    # This would call your existing run_audio_chatbot_v2 function
    
    # Note: The actual function calls are commented out since these are external functions
    # In real implementation, you would uncomment these lines
    # run_audio_chatbot_v2(start_trigger='1', stop_trigger='2', response_generator=prediction_chatbot_handler)
    
    print("\nChatbot integration complete!")

# 11. Main Execution Function
def run_prediction_pipeline(data_path='data_set.csv', retrain=False):
    """
    Run the complete prediction pipeline
    
    Parameters:
    -----------
    data_path : str
        Path to the input CSV file
    retrain : bool
        Whether to retrain the models or load existing ones
        
    Returns:
    --------
    DataFrame with predictions
    """
    # Load data
    df = load_data(data_path)
    
    if retrain:
        # Preprocess data
        X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features = preprocess_data(df)
        
        # Train base models
        trained_models, cv_results = train_base_models(X_train, y_train, preprocessor)
        
        # Evaluate base models
        evaluation = evaluate_models(trained_models, X_test, y_test)
        
        # Build stacking model
        stacking_model, stacking_evaluation = build_stacking_model(
            trained_models, X_train, y_train, X_test, y_test
        )
        
        # Save pipeline
        saved_files = save_pipeline(
            trained_models, 
            stacking_model, 
            preprocessor, 
            numerical_features, 
            categorical_features, 
            cv_results, 
            evaluation, 
            stacking_evaluation
        )
        
        # Make predictions
        pipeline = {
            'base_models': trained_models,
            'stacking_model': stacking_model,
            'preprocessor': preprocessor,
            'numerical_features': numerical_features,
            'categorical_features': categorical_features
        }
        predictions_df = make_predictions(df, pipeline, return_all_predictions=True)
        
    else:
        # Load existing pipeline
        pipeline = load_pipeline()
        
        # Make predictions
        predictions_df = make_predictions(df, pipeline, return_all_predictions=True)
    
    # Integrate with chatbot
    integrate_with_chatbot(predictions_df)
    
    return predictions_df

# Run the complete pipeline
if __name__ == "__main__":
    predictions_df = run_prediction_pipeline(retrain=True)
    print("\nPipeline execution complete!")