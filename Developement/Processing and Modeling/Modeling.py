import os
import subprocess
import sys
import time
from pathlib import Path
def check_and_install_requirements():
    print("Checking and installing required packages...")
    if not os.path.exists('requirements.txt'):
        with open('requirements.txt', 'w') as f:
            f.write("""pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
joblib>=1.0.0
tqdm>=4.60.0
textblob>=0.15.3
scipy>=1.7.0
pydub>=0.25.1
pyaudio>=0.2.11
SoundDevice>=0.4.1
pyttsx3>=2.90
requests>=2.25.1
python-dotenv>=0.19.0
google-generativeai>=0.0.1
groq>=0.5.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
optuna>=2.10.0
shap>=0.40.0
""")
    if sys.platform.startswith('linux'):
        try:
            subprocess.check_call(['apt-get', 'update'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.check_call(['apt-get', 'install', '-y', 'portaudio19-dev', 'python3-pyaudio'], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print("Warning: Could not install system dependencies. You may need sudo privileges.")
            print("Try running: sudo apt-get install portaudio19-dev python3-pyaudio")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("All required packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        print("Please install the required packages manually using:")
        print("pip install -r requirements.txt")
check_and_install_requirements()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import random
import joblib
import re
from tqdm import tqdm
import time
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

print("Libraries imported successfully!")
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
def load_data(file_path, sample_fraction=0.001, random_state=42):
    print(f"Loading data from {file_path}...")
    file_extension = file_path.lower().split('.')[-1]
    try:
        if file_extension == 'csv':
            df = pd.read_csv(file_path)
        elif file_extension == 'json':
            df = pd.read_json(file_path)
        elif file_extension == 'jsonl':
            df = pd.read_json(file_path, lines=True)
        elif file_extension in ['parquet', 'pqt']:
            df = pd.read_parquet(file_path)
        else:
            print(f"Unsupported file format: {file_extension}")
            print("Supported formats: CSV, JSON, JSONL, Parquet")
            return None
        print(f"Original dataset shape: {df.shape}")
        if sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=random_state)
            print(f"Sampled {sample_fraction*100:.0f}% of data")
            print(f"Sampled dataset shape: {df.shape}")
        print("\nColumn types:")
        print(df.dtypes)
        print("\nSample data:")
        print(df.head())
        print("\nMissing values per column:")
        print(df.isna().sum())
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None
df = load_data('data_set.csv')
def preprocess_data(df, target_column='log_price', test_size=0.2, random_state=42):
    print("Starting data preprocessing...")
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['quarter'] = df['timestamp'].dt.quarter
        df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['days_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
        df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    if 'text' in df.columns:
        df['text_length'] = df['text'].fillna('').apply(len)
        df['word_count'] = df['text'].fillna('').apply(lambda x: len(x.split()))
        df['avg_word_length'] = df['text'].fillna('').apply(lambda x: np.mean([len(w) for w in x.split()] or [0]))
        df['caps_ratio'] = df['text'].fillna('').apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
        
    if 'sentiment_score' not in df.columns and 'text' in df.columns:
        try:
            from textblob import TextBlob
            df['sentiment_score'] = df['text'].fillna('').apply(lambda x: TextBlob(x).sentiment.polarity)
            df['sentiment_magnitude'] = df['text'].fillna('').apply(lambda x: abs(TextBlob(x).sentiment.polarity))
            df['subjectivity'] = df['text'].fillna('').apply(lambda x: TextBlob(x).sentiment.subjectivity)
            df['sentiment_category'] = pd.cut(
                df['sentiment_score'], 
                bins=[-1.1, -0.3, 0.3, 1.1], 
                labels=['negative', 'neutral', 'positive']
                )
        except:
            print("TextBlob not available, skipping sentiment analysis")
    if 'title' in df.columns:
            df['title_length'] = df['title'].fillna('').apply(len)
            df['has_number_in_title'] = df['title'].fillna('').apply(lambda x: any(c.isdigit() for c in x)).astype(int)
            df['numbers_in_title'] = df['title'].fillna('').apply(lambda x: len(re.findall(r'\d+', x)))
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and col != target_column:
            if df[col].min() >= 0 and df[col].skew() > 1.5:
                skew_name = f'{col}_log'
                df[skew_name] = np.log1p(df[col])
    if 'rating' in df.columns:
        df['rating_squared'] = df['rating'] ** 2
        df['high_rating'] = (df['rating'] >= 4.0).astype(int)
        if 'rating' in df.columns and 'total_reviews' in df.columns:
            df['rating_reviews_ratio'] = df['rating'] * np.log1p(df['total_reviews'])
    if 'main_category' in df.columns:
        category_counts = df['main_category'].value_counts()
        df['category_frequency'] = df['main_category'].map(category_counts)
        df['category_frequency_norm'] = df['category_frequency'] / df['category_frequency'].max()
    
    categorical_features = [col for col in ['domain', 'main_category', 'sentiment_category']
                           if col in df.columns]
    numerical_features = [col for col in df.columns 
                          if df[col].dtype in ['int64', 'float64'] 
                          and col != target_column
                          and col not in categorical_features]
    print(f"Using categorical features: {categorical_features}")
    print(f"Using numerical features: {numerical_features}")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    X = df[numerical_features + categorical_features]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features
X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features = preprocess_data(df)
def train_base_models(X_train, y_train, preprocessor, cv=5):
    print("\nTraining base models with cross-validation...")
    random_seeds = [random.randint(1, 10000) for _ in range(6)]
    base_models = {
        'random_forest': RandomForestRegressor(random_state=random_seeds[0]),
        'gradient_boosting': GradientBoostingRegressor(random_state=random_seeds[1]),
        'xgboost': XGBRegressor(random_state=random_seeds[2], verbosity=0,enable_categorical=True),
        'lightgbm': LGBMRegressor(random_state=random_seeds[3], verbose=-1),
        'catboost': CatBoostRegressor(random_state=random_seeds[4], verbose=0),
        'ridge': Ridge(random_state=random_seeds[5])
    }
    param_grids = {
        'random_forest': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        },
        'gradient_boosting': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0]
        },
        'xgboost': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 5, 7],
            'model__colsample_bytree': [0.7, 0.9]
        },
        'lightgbm': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__num_leaves': [31, 50, 70],
            'model__reg_alpha': [0.0, 0.1, 0.5]
        },
        'catboost': {
            'model__iterations': [100, 200],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__depth': [4, 6, 8]
        },
        'ridge': {
            'model__alpha': [0.01, 0.1, 1.0, 10.0],
            'model__solver': ['auto', 'svd']
        }
    }
    total_models = len(base_models)
    with tqdm(total=total_models, desc="Overall Model Training Progress", position=0) as master_pbar:
        trained_models = {}
        cv_results = {}
        for name, model in base_models.items():
            print(f"\nTraining {name}...")
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            total_param_combinations = 1
            for param_values in param_grids[name].values():
                total_param_combinations *= len(param_values)
            total_iterations = total_param_combinations * cv
            model_pbar = tqdm(total=total_iterations,
                             desc=f"{name.capitalize()} Training",
                             position=1,
                             leave=False)
            def progress_callback(iteration, total, status=''):
                model_pbar.update(1)
            class VerboseCallback:
                def __init__(self, estimator_name, verbose=0):
                    self.name = estimator_name
                    self.verbose = verbose
                    self.n_iter = 0
                def __call__(self, estimator, fold_idx, param_idx):
                    self.n_iter += 1
                    if self.verbose > 0:
                        progress_callback(self.n_iter, None)
                    return estimator
            grid_search = GridSearchCV(
                pipeline,
                param_grids[name],
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            iter_count = 0
            for params in grid_search.param_grid:
                pass
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            train_time = time.time() - start_time
            model_pbar.update(total_iterations - model_pbar.n)
            model_pbar.close()
            trained_models[name] = grid_search.best_estimator_
            cv_results[name] = {
                'best_score': -grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'training_time': train_time
            }
            print(f"Best {name} score (MSE): {-grid_search.best_score_:.4f}")
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Training time: {train_time:.2f} seconds")
            master_pbar.update(1)
    print("\nModel Training Summary:")
    for name, results in cv_results.items():
        print(f"{name.capitalize()}: MSE = {results['best_score']:.4f}, Time = {results['training_time']:.2f}s")
    return trained_models, cv_results
trained_models, cv_results = train_base_models(X_train, y_train, preprocessor)

def generate_chat_response_v2(prompt):
    import google.generativeai as genai
    genai.configure(api_key="AIzaSyD6VxooY8ZAop729dNk0NRFTHxBX380Lr8")
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text
def evaluate_models(models, X_test, y_test):
    print("\nEvaluating models on test data...")
    evaluation = {}
    predictions = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
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
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    actual_indices = range(len(y_test))
    plt.scatter(actual_indices[:100], y_test.iloc[:100], color='black', alpha=0.7, label='Actual', s=40)
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
    for i, (name, y_pred) in enumerate(predictions.items()):
        plt.scatter(actual_indices[:100], y_pred[:100], color=colors[i % len(colors)], 
                   alpha=0.5, label=f'{name.capitalize()}', s=30)
    plt.title('Actual vs Predicted log_price (First 100 samples)', fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('log_price', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 1, 2)
    error_data = []
    labels = []
    for name, y_pred in predictions.items():
        error_data.append(y_test - y_pred)
        labels.append(name.capitalize())
    sns.violinplot(data=error_data, palette='viridis')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Prediction Error Distribution by Model', fontsize=14)
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.ylabel('Prediction Error (Actual - Predicted)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    for i, (name, model) in enumerate(models.items(), 1):
        y_pred = predictions[name]
        rmse = evaluation[name]['RMSE']
        r2 = evaluation[name]['R2']
        fig = plt.figure(figsize=(16, 6))
        plt.subplot(1, 3, 1)
        hexbin = plt.hexbin(y_test, y_pred, gridsize=50, cmap='viridis', mincnt=1)
        plt.colorbar(hexbin, label='Count')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.text(0.05, 0.95, f"RMSE: {rmse:.4f}\nR²: {r2:.4f}",
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.xlabel('Actual log_price')
        plt.ylabel('Predicted log_price')
        plt.title(f'{name.capitalize()} - Prediction Performance')
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 3, 2)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5, color='navy')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted log_price')
        plt.ylabel('Residuals')
        plt.title(f'Residuals Analysis')
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 3, 3)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        plt.tight_layout()
        plt.suptitle(f'{name.capitalize()} Model Evaluation', fontsize=16, y=1.05)
        plt.show()
    plt.figure(figsize=(12, 8))
    metrics_df = pd.DataFrame(
        {name: [values['RMSE'], values['MAE'], values['R2']] for name, values in evaluation.items()},
        index=['RMSE', 'MAE', 'R2']
    ).T
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    metrics_df['RMSE'].sort_values().plot(kind='barh', ax=axes[0], color='coral')
    axes[0].set_title('RMSE by Model (Lower is Better)', fontsize=14)
    axes[0].grid(axis='x', alpha=0.3)
    metrics_df['MAE'].sort_values().plot(kind='barh', ax=axes[1], color='skyblue')
    axes[1].set_title('MAE by Model (Lower is Better)', fontsize=14)
    axes[1].grid(axis='x', alpha=0.3)
    metrics_df['R2'].sort_values(ascending=False).plot(kind='barh', ax=axes[2], color='lightgreen')
    axes[2].set_title('R² Score by Model (Higher is Better)', fontsize=14)
    axes[2].grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    return evaluation
evaluation = evaluate_models(trained_models, X_test, y_test)
def plot_residuals(y_test, y_pred, model_name):
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5, color='navy')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{model_name} - Residuals vs Predicted')
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
def plot_feature_importance(models, preprocessor, numerical_features, categorical_features):
    print("\nAnalyzing feature importance...")
    feature_names = []
    feature_names.extend(numerical_features)
    if hasattr(preprocessor, 'transformers_'):
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'cat' and hasattr(transformer, 'named_steps') and 'onehot' in transformer.named_steps:
                onehot = transformer.named_steps['onehot']
                if hasattr(onehot, 'get_feature_names_out'):
                    cat_features = onehot.get_feature_names_out(cols)
                    feature_names.extend(cat_features)
                elif hasattr(onehot, 'categories_'):
                    for i, categories in enumerate(onehot.categories_):
                        for cat in categories:
                            feature_names.append(f"{cols[i]}_{cat}")
    tree_models = {}
    for name, model in models.items():
        if hasattr(model, 'named_steps') and hasattr(model.named_steps.get('model', None), 'feature_importances_'):
            tree_models[name] = model.named_steps['model']
        elif hasattr(model, 'feature_importances_'):
            tree_models[name] = model
    if not tree_models:
        print("No tree-based models found with feature_importances_ attribute")
        return
    all_importances = {}
    for name, model in tree_models.items():
        importances = model.feature_importances_
        n_features = min(len(importances), len(feature_names))
        importances = importances[:n_features]
        feature_names_use = feature_names[:n_features]
        all_importances[name] = {feature_names_use[i]: importances[i] 
                               for i in range(len(importances))}
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        indices = np.argsort(importances)[-15:]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(indices)))
        ax1.barh(range(len(indices)), importances[indices], align='center', color=colors)
        ax1.set_yticks(range(len(indices)))
        ax1.set_yticklabels([feature_names_use[i] for i in indices])
        ax1.set_xlabel('Feature Importance Score')
        ax1.set_title(f'Top 15 Features - {name}')
        ax1.grid(axis='x', alpha=0.3)
        top10_indices = np.argsort(importances)[-10:]
        top10_importances = importances[top10_indices]
        top10_names = [feature_names_use[i] for i in top10_indices]
        total_importance = sum(top10_importances)
        percentages = [100 * imp/total_importance for imp in top10_importances]
        wedges, texts, autotexts = ax2.pie(
            percentages, 
            labels=[f"{name} ({pct:.1f}%)" for name, pct in zip(top10_names, percentages)],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'alpha': 0.7},
            textprops={'fontsize': 9}
        )
        plt.setp(autotexts, size=8, weight="bold")
        ax2.set_title(f'Top 10 Feature Importance Distribution - {name}')
        plt.tight_layout()
        plt.suptitle(f'{name.capitalize()} - Feature Importance Analysis', fontsize=16, y=1.05)
        plt.subplots_adjust(top=0.85)
        plt.show()
    if len(tree_models) > 1:
        all_features = set()
        for model_importances in all_importances.values():
            all_features.update(model_importances.keys())
        feature_avg_importance = {}
        for feature in all_features:
            values = [model_imp.get(feature, 0) for model_imp in all_importances.values()]
            feature_avg_importance[feature] = np.mean(values)
        top_features = sorted(feature_avg_importance.items(), 
                             key=lambda x: x[1], reverse=True)[:10]
        top_feature_names = [f[0] for f in top_features]
        comparison_data = []
        for feature in top_feature_names:
            feature_data = [model_imp.get(feature, 0) for name, model_imp 
                           in all_importances.items()]
            comparison_data.append(feature_data)
        plt.figure(figsize=(14, 8))
        x = np.arange(len(top_feature_names))
        width = 0.8 / len(tree_models)
        for i, (name, _) in enumerate(tree_models.items()):
            offsets = [p[i] for p in comparison_data]
            plt.bar(x + i*width - 0.4 + width/2, offsets, width, label=name)
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title('Feature Importance Comparison Across Models', fontsize=14)
        plt.xticks(x, top_feature_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.show()
def build_stacking_model(base_models, X_train, y_train, X_test, y_test):
    print("\nBuilding stacking regressor model...")
    estimators = [(name, model) for name, model in base_models.items()]
    
    # Convert DataFrame to NumPy array before feature selection
    X_train_fixed = X_train.copy()
    X_test_fixed = X_test.copy()
    
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train_fixed[col] = X_train_fixed[col].astype('category')
        X_test_fixed[col] = X_test_fixed[col].astype('category')
    
    # Remove the feature selection part that's causing issues
    # Use all features directly instead
    selected_columns = X_train_fixed.columns.tolist()
    
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=XGBRegressor(random_state=42, learning_rate=0.05, n_estimators=100, enable_categorical=True),
        cv=5,
        n_jobs=-1
    )
    
    # Use the DataFrame directly for the stacking regressor
    stacking_regressor.fit(X_train_fixed, y_train)
    y_pred = stacking_regressor.predict(X_test_fixed)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mean_percentage_error = np.mean(np.abs((np.exp(y_test) - np.exp(y_pred)) / np.exp(y_test))) * 100
    
    evaluation = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MPE': mean_percentage_error
    }
    
    print(f"Stacking Regressor performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")
    
    # Store the selected columns for later use without feature selector
    stacking_regressor.selected_columns = selected_columns
    
    return stacking_regressor, evaluation
def plot_shap_feature_importance(model, X_test, feature_names):
    import shap
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        final_model = model.named_steps['model']
    else:
        final_model = model
    if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
        X_processed = model.named_steps['preprocessor'].transform(X_test)
    else:
        X_processed = X_test
        
    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_processed)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_processed, feature_names=feature_names)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_processed, feature_names=feature_names, 
                          max_display=20, plot_type="bar")
        plt.title("Top 20 Features by SHAP Importance")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not generate SHAP plots: {e}")
def save_pipeline(base_models, stacking_model, preprocessor,
                 numerical_features, categorical_features,
                 cv_results, evaluation, stacking_evaluation):
    print("\nSaving machine learning pipeline components...")
    saved_files = {}
    for name, model in base_models.items():
        model_path = f'models/{name}_model.pkl'
        joblib.dump(model, model_path)
        saved_files[f'{name}_model'] = model_path
    stacking_path = 'models/stacking_model.pkl'
    joblib.dump(stacking_model, stacking_path)
    saved_files['stacking_model'] = stacking_path
    preprocessor_path = 'models/preprocessor.pkl'
    joblib.dump(preprocessor, preprocessor_path)
    saved_files['preprocessor'] = preprocessor_path
    features = {
        'numerical_features': numerical_features,
        'categorical_features': categorical_features
    }
    features_path = 'models/feature_lists.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)
    saved_files['features'] = features_path
    results = {
        'cv_results': cv_results,
        'evaluation': evaluation,
        'stacking_evaluation': stacking_evaluation
    }
    results_path = 'models/evaluation_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    saved_files['results'] = results_path
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
stacking_model, stacking_evaluation = build_stacking_model(
    trained_models, X_train, y_train, X_test, y_test
)
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
def load_pipeline():
    print("\nLoading machine learning pipeline components...")
    pipeline = {}
    with open('models/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    pipeline['metadata'] = metadata
    base_models = {}
    for name in metadata['model_names']:
        model_path = f'models/{name}_model.pkl'
        base_models[name] = joblib.load(model_path)
    pipeline['base_models'] = base_models
    pipeline['stacking_model'] = joblib.load('models/stacking_model.pkl')
    pipeline['preprocessor'] = joblib.load('models/preprocessor.pkl')
    with open('models/feature_lists.pkl', 'rb') as f:
        features = pickle.load(f)
    pipeline['numerical_features'] = features['numerical_features']
    pipeline['categorical_features'] = features['categorical_features']
    with open('models/evaluation_results.pkl', 'rb') as f:
        results = pickle.load(f)
    pipeline['cv_results'] = results['cv_results']
    pipeline['evaluation'] = results['evaluation']
    pipeline['stacking_evaluation'] = results['stacking_evaluation']
    print("Pipeline components loaded successfully!")
    return pipeline
def make_predictions(df, pipeline, return_all_predictions=False):
    print("\nPreparing data for prediction...")
    numerical_features = pipeline['numerical_features']
    categorical_features = pipeline['categorical_features']
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['dayofweek'] = df['timestamp'].dt.dayofweek
    if 'text' in df.columns and 'text_length' not in df.columns:
        df['text_length'] = df['text'].fillna('').apply(len)
    X = df[numerical_features + categorical_features].copy()
    print("Making predictions...")
    predictions = {}
    for name, model in pipeline['base_models'].items():
        predictions[f'{name}_prediction'] = model.predict(X)
    stacking_model = pipeline['stacking_model']
    if hasattr(stacking_model, 'selected_columns'):
        X_selected = X[stacking_model.selected_columns]
        predictions['stacking_prediction'] = stacking_model.predict(X_selected)
    else:
        predictions['stacking_prediction'] = stacking_model.predict(X)
    results_df = df.copy()
    for name, preds in predictions.items():
        results_df[name] = preds

    if 'log_price' in results_df.columns:
        for name in predictions.keys():
            orig_price_col = name.replace('prediction', 'orig_price')
            results_df[orig_price_col] = np.exp(results_df[name])
    
    print("Predictions completed!")
    
    if return_all_predictions:
        return results_df
    else:
        selected_cols = list(df.columns) + ['stacking_prediction']
        if 'log_price' in results_df.columns:
            selected_cols.append('stacking_orig_price')
        return results_df[selected_cols]
    
def create_knowledge_base(df):
    print("\nCreating comprehensive knowledge base from dataset...")
    df['kb_text'] = df.apply(
        lambda row: f"Product ID: {row.get('asin', 'Unknown')} | "
                  + f"Category: {row.get('main_category', 'Unknown')} | "
                  + f"Rating: {row.get('rating', 'Unknown')}/5 | "
                  + f"Price: ${np.exp(row['log_price']):.2f} | "
                  + f"Reviews: {row.get('total_reviews', 'Unknown')} | "
                  + f"Sentiment: {row.get('sentiment_score', 'Unknown')} | "
                  + f"Year: {row.get('year', 'Unknown')} | "
                  + f"Product Title: {row.get('title', 'Unknown')} | "
                  + f"Description: {row.get('text', '')[:500]}",
        axis=1
    )
    tfidf_vectorizer = TfidfVectorizer(
        max_features=15000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.85
    )
    doc_vectors = tfidf_vectorizer.fit_transform(df['kb_text'])
    metadata = {
        'feature_stats': {
            'price_mean': np.exp(df['log_price']).mean(),
            'price_median': np.exp(df['log_price']).median(),
            'price_range': (np.exp(df['log_price']).min(), np.exp(df['log_price']).max()),
            'top_categories': df['main_category'].value_counts().head(5).to_dict(),
            'rating_distribution': df['rating'].value_counts().to_dict() if 'rating' in df.columns else None,
            'product_count': len(df)
        }
    }
    kb_path = 'models/knowledge_base.pkl'
    joblib.dump((tfidf_vectorizer, doc_vectors, df['kb_text'].tolist(), metadata), kb_path)
    print(f"Enhanced knowledge base created with {doc_vectors.shape[0]} documents and {doc_vectors.shape[1]} features")
    return tfidf_vectorizer, doc_vectors, df['kb_text'].tolist(), metadata

def retrieve_relevant_documents(query, vectorizer, doc_vectors, documents, top_n=7):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    query_terms = set(query.lower().split())
    boosted_similarities = similarities.copy()
    for i, doc in enumerate(documents):
        doc_lower = doc.lower()
        term_matches = sum(1 for term in query_terms if term in doc_lower)
        phrase_boost = 0
        for phrase in [' '.join(query_terms), query.lower()]:
            if len(phrase) > 3 and phrase in doc_lower:
                phrase_boost += 0.2
        if term_matches > 0:
            boosted_similarities[i] += 0.15 * np.log1p(term_matches) + phrase_boost
    top_indices = boosted_similarities.argsort()[-top_n:][::-1]
    top_docs = [(documents[i], boosted_similarities[i], i) for i in top_indices]
    return top_docs

def generate_prediction_response(query, df_predictions, knowledge_base):
    vectorizer, doc_vectors, documents, metadata = knowledge_base
    relevant_docs = retrieve_relevant_documents(query, vectorizer, doc_vectors, documents, top_n=7)
    query_lower = query.lower()
    intents = {
        'price_focused': any(term in query_lower for term in 
                           ['price', 'cost', 'expensive', 'cheap', 'worth', 'value', 'affordable', 'budget', 'money']),
        'feature_focused': any(term in query_lower for term in 
                             ['feature', 'specification', 'quality', 'rating', 'review', 'characteristic']),
        'comparison_focused': any(term in query_lower for term in 
                                ['compare', 'versus', 'vs', 'better', 'difference', 'similar', 'alternative']),
        'category_focused': any(term in query_lower for term in 
                              ['category', 'type', 'group', 'kind', 'department']),
        'specific_product': any(term in query_lower for term in 
                              ['specific', 'exact', 'particular', 'this product', 'item'])
    }
    context_items = []
    prediction_examples = []
    for doc, score, idx in relevant_docs:
        try:
            product_id = doc.split("Product ID: ")[1].split(" |")[0]
            product_pred = df_predictions[df_predictions['asin'] == product_id]
            if not product_pred.empty and 'stacking_prediction' in product_pred.columns:
                predicted_price = np.exp(product_pred['stacking_prediction'].iloc[0])
                actual_price = np.exp(product_pred['log_price'].iloc[0]) if 'log_price' in product_pred.columns else None
                prediction_examples.append({
                    'id': product_id,
                    'predicted': predicted_price,
                    'actual': actual_price,
                    'category': product_pred['main_category'].iloc[0] if 'main_category' in product_pred.columns else 'Unknown',
                    'similarity': score
                })
                if actual_price:
                    price_info = f"Actual price: ${actual_price:.2f} | Predicted price: ${predicted_price:.2f}"
                else:
                    price_info = f"Predicted price: ${predicted_price:.2f}"
                context_item = f"Product (relevance: {score:.2f}):\n{doc}\n{price_info}"
            else:
                context_item = f"Product (relevance: {score:.2f}):\n{doc}"
        except Exception as e:
            context_item = f"Product (relevance: {score:.2f}):\n{doc}\n(No prediction available)"
        context_items.append(context_item)
    context = "\n\n".join(context_items)
    insights = []
    insights.append(f"Product Database: {metadata['feature_stats']['product_count']} products analyzed")
    if intents['price_focused']:
        if prediction_examples:
            relevant_prices = [ex['predicted'] for ex in prediction_examples]
            insights.append(f"For products similar to your query:")
            insights.append(f"- Average predicted price: ${np.mean(relevant_prices):.2f}")
            insights.append(f"- Price range: ${min(relevant_prices):.2f} - ${max(relevant_prices):.2f}")
        insights.append(f"Overall market insights:")
        insights.append(f"- Market average price: ${metadata['feature_stats']['price_mean']:.2f}")
        insights.append(f"- Market median price: ${metadata['feature_stats']['price_median']:.2f}")
        insights.append(f"- Full price range: ${metadata['feature_stats']['price_range'][0]:.2f} - ${metadata['feature_stats']['price_range'][1]:.2f}")
    if intents['category_focused']:
        insights.append("Top product categories by volume:")
        for cat, count in metadata['feature_stats']['top_categories'].items():
            insights.append(f"- {cat}: {count} products")
    if intents['feature_focused'] or intents['comparison_focused']:
        try:
            with open('models/model_metadata.pkl', 'rb') as f:
                model_metadata = pickle.load(f)
            top_num_features = model_metadata.get('numerical_features', [])[:5]
            top_cat_features = model_metadata.get('categorical_features', [])[:3]
            insights.append("Key factors affecting product prices:")
            insights.append(f"- Important numerical factors: {', '.join(top_num_features)}")
            insights.append(f"- Important categorical factors: {', '.join(top_cat_features)}")
            if intents['comparison_focused'] and prediction_examples:
                categories = set(ex['category'] for ex in prediction_examples if ex['category'] != 'Unknown')
                if len(categories) > 1:
                    insights.append("\nPrice comparison between relevant categories:")
                    for category in categories:
                        cat_examples = [ex for ex in prediction_examples if ex['category'] == category]
                        if cat_examples:
                            avg_price = np.mean([ex['predicted'] for ex in cat_examples])
                            insights.append(f"- {category}: ${avg_price:.2f} average predicted price")
        except:
            pass
    if metadata['feature_stats']['rating_distribution'] and (intents['feature_focused'] or intents['comparison_focused']):
        insights.append("\nRating distribution across products:")
        for rating, count in sorted(metadata['feature_stats']['rating_distribution'].items()):
            if isinstance(rating, (int, float)):
                insights.append(f"- {rating} stars: {count} products")
    insights_text = "\n".join(insights)
    prompt_for_gemini = f"""
    You are a helpful AI assistant specializing in product pricing analysis and recommendations.
    Based on the user's query: "{query}"
    
    DATA INSIGHTS:
    {insights_text}
    
    RELEVANT PRODUCT INFORMATION:
    {context}
    
    Please provide a helpful, conversational response that directly addresses the user's query.
    Focus on providing valuable insights about product pricing, features, and comparisons based on the data.
    Keep your response concise, factual, and specifically tailored to what the user is asking about.
    """
    return prompt_for_gemini

def prediction_chatbot_handler(query):
    processed_query = query.strip()
    if "help" in processed_query.lower() or "what can you do" in processed_query.lower():
        return """
        I can help you with:
        1. Price predictions for products in our dataset
        2. Information about product features and their impact on price
        3. Comparisons between different product categories
        4. Details about specific products if they're in our knowledge base
        5. Market insights and pricing trends
            
        Try asking questions like:
        - "What's the average predicted price for electronics?"
        - "How do features affect the price of kitchen products?"
        - "Compare prices between books and electronics"
        - "Tell me about products with high ratings"
        - "What factors most influence product pricing?"
        """
    response = generate_prediction_response(processed_query, df_predictions, knowledge_base)
    return response
def process_new_data():
    print("\nNew Data Prediction Mode")
    print("Supported file formats: CSV, JSON, JSONL, Parquet")
    print("Enter data for prediction (file path or 'manual' for manual entry):")
    
    # Load the pipeline to get required columns
    pipeline = load_pipeline()
    numerical_features = pipeline['numerical_features']
    categorical_features = pipeline['categorical_features']
    required_columns = numerical_features + categorical_features
    
    print(f"\nRequired columns for prediction:")
    print(f"Numerical: {', '.join(numerical_features)}")
    print(f"Categorical: {', '.join(categorical_features)}")
    
    data_input = input("\nEnter file path or 'manual' for manual entry: ").strip()
    
    if data_input.lower() == 'manual':
        print("\nEnter data manually (leave blank to use default value):")
        data_dict = {}
        
        print("\nNumerical features:")
        for feature in numerical_features:
            value = input(f"{feature} (numeric): ").strip()
            if value:
                try:
                    data_dict[feature] = float(value)
                except ValueError:
                    print(f"Invalid numeric value for {feature}, using median value instead.")
        
        print("\nCategorical features:")
        for feature in categorical_features:
            value = input(f"{feature} (text): ").strip()
            if value:
                data_dict[feature] = value
                
        new_data = pd.DataFrame([data_dict])
        for feature in required_columns:
            if feature not in new_data.columns:
                new_data[feature] = None
    else:
        try:
            # Detect file format from extension
            file_extension = data_input.lower().split('.')[-1]
            
            if file_extension == 'csv':
                new_data = pd.read_csv(data_input)
            elif file_extension == 'json':
                new_data = pd.read_json(data_input)
            elif file_extension == 'jsonl':
                new_data = pd.read_json(data_input, lines=True)
            elif file_extension in ['parquet', 'pqt']:
                new_data = pd.read_parquet(data_input)
            else:
                print(f"Unsupported file format: {file_extension}")
                print("Supported formats: CSV, JSON, JSONL, Parquet")
                return
                
            # Check for required columns
            missing_columns = [col for col in required_columns if col not in new_data.columns]
            if missing_columns:
                print(f"Warning: Missing required columns: {', '.join(missing_columns)}")
                print("These columns will be initialized with null values.")
                for col in missing_columns:
                    new_data[col] = None
                    
        except Exception as e:
            print(f"Error loading data: {e}")
            return
            
    print("\nProcessing new data for prediction...")
    predictions = make_predictions(new_data, pipeline, return_all_predictions=True)
    
    print("\nPrediction Results:")
    if 'stacking_prediction' in predictions.columns:
        for i, row in predictions.iterrows():
            orig_price = np.exp(row['stacking_prediction'])
            print(f"Item {i+1} - Predicted price: ${orig_price:.2f}")
            
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.hist(np.exp(predictions['stacking_prediction']), bins=20, alpha=0.7)
            plt.title('Distribution of Predicted Prices')
            plt.xlabel('Predicted Price ($)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.show()
        except Exception as e:
            print(f"Error generating visualization: {e}")
    else:
        print("No predictions generated. Check if the input data contains required features.")
        
    return predictions
pipeline = load_pipeline()
df_predictions = make_predictions(df, pipeline, return_all_predictions=True)
knowledge_base = create_knowledge_base(df)
print("\nAvailable interaction modes:")
print("1. Text-to-Text Chat")
print("2. Speech-to-Speech Chat")
print("3. New Data Prediction")
while True:
    mode = input("\nSelect interaction mode (1, 2, or 3) or 'exit' to quit: ")
    if mode.lower() in ['exit', 'quit']:
        print("Exiting program. Goodbye!")
        break
    if mode == "1":
        print("\nStarting text chatbot...")
        def run_text_chatbot_v2(response_generator):
            intro_message = """
            Welcome to the Price Forecasting Assistant! 🤖📊
            
            I can help you with:
            - Analyzing product pricing data and trends
            - Making price predictions for items in our dataset
            - Answering questions about market insights and price factors
            - Processing new data for forecasting
            
            I combine my general knowledge with specific insights from our trained machine learning models.
            
            Type 'help' to see more examples of what you can ask, or 'exit' to quit.
            """
            
            print(intro_message)
            
            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("Ending chat session.")
                    break
                    
                system_response = response_generator(user_input)
                try:
                    final_response = generate_chat_response_v2(
                        f"You are an AI price forecasting assistant trained on product data. You help with price predictions and market insights. Respond to this query in a helpful, concise way: {system_response}"
                    )
                    print(f"\nBot: {final_response}")
                except Exception as e:
                    print(f"\nBot: {system_response}")
                    print(f"\n(Note: External API call failed: {str(e)})")
                    
                continue_chat = input("\nContinue chatting? (y/n): ").strip().lower()
                if continue_chat != 'y':
                    print("Ending chat session.")
                    break
        run_text_chatbot_v2(response_generator=prediction_chatbot_handler)
    elif mode == "2":
        print("\nStarting audio chatbot...")
        print("Use '1' to start and '2' to stop recording.")
        import threading
        import sounddevice as sd
        import numpy as np
        from scipy.io.wavfile import write
        from pydub import AudioSegment
        import requests
        import pyttsx3
        import google.generativeai as genai

        # Configure Gemini API once at the start
        genai.configure(api_key="AIzaSyD6VxooY8ZAop729dNk0NRFTHxBX380Lr8")
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        
        def generate_chat_response(prompt):
            try:
                response = gemini_model.generate_content(prompt)
                return response.text if hasattr(response, "text") else str(response)
            except Exception as e:
                print("Gemini API error:", e)
                return "I'm sorry, I encountered an error processing your request."
        
        def run_audio_chatbot_v2(start_trigger='1', stop_trigger='2', response_generator=None):
            chat_continue = True
            while chat_continue:
                fs = 44100
                file_path = "user_recording.wav"
                mp3_path = file_path.replace(".wav", ".mp3")
                print(f"Press {start_trigger} to start recording...")
                start_input = input().strip()
                if start_input != start_trigger:
                    print("Invalid start input. Exiting.")
                    return
                
                recording_flag = True
                audio_frames = []
                
                def handle_stop():
                    nonlocal recording_flag
                    print(f"\nPress {stop_trigger} to stop recording...")
                    while True:
                        user_input = input().strip()
                        if user_input == stop_trigger:
                            recording_flag = False
                            break
                
                stop_thread = threading.Thread(target=handle_stop)
                stop_thread.start()
                
                try:
                    with sd.InputStream(samplerate=fs, channels=1, callback=lambda indata, frames, time, status: audio_frames.append(indata.copy())):
                        while recording_flag:
                            sd.sleep(1)
                except Exception as e:
                    print(f"Error during recording: {e}")
                    return
                
                stop_thread.join()
                
                if not audio_frames:
                    print("No audio recorded.")
                    return
                
                audio_array = np.concatenate(audio_frames, axis=0)
                write(file_path, fs, audio_array)
                audio = AudioSegment.from_wav(file_path)
                audio.export(mp3_path, format="mp3")
                
                tr_api_key = "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"
                transcribe_url = "https://api.groq.com/openai/v1/audio/transcriptions"
                headers = {"Authorization": f"Bearer {tr_api_key}"}
                
                with open(mp3_path, "rb") as file:
                    files = {"file": file}
                    data = {"model": "whisper-large-v3", "language": "en"}
                    response = requests.post(transcribe_url, headers=headers, files=files, data=data)
                    transcription = response.json()
                
                if 'text' not in transcription:
                    print("Transcription failed:", transcription.get("error", transcription))
                    return
                
                print("\nTranscription:\n", transcription["text"])
                
                if any(exit_word in transcription["text"].lower() for exit_word in ["exit", "quit", "bye", "goodbye"]):
                    print("Exit command detected. Ending chat session.")
                    break
                
                # Process the user query with the response generator if provided
                user_query = transcription["text"]
                if response_generator:
                    prediction_response = response_generator(user_query)
                    prompt = f"Respond to this query about price predictions: {prediction_response}"
                else:
                    prompt = user_query
                
                # Use the simplified Gemini response function
                response_text = generate_chat_response(prompt)
                
                print("\nChatbot response:\n", response_text)
                
                engine = pyttsx3.init()
                try:
                    engine.endLoop()
                except:
                    pass
                engine.say(response_text)
                engine.runAndWait()
                
                continue_response = input("\nContinue with speech chat? (y/n): ").strip().lower()
                chat_continue = (continue_response == 'y')
                if not chat_continue:
                    print("Ending speech chat session.")
        
        run_audio_chatbot_v2(start_trigger='1', stop_trigger='2', 
                        response_generator=prediction_chatbot_handler)
    elif mode == "3":
        new_predictions = process_new_data()
        continue_session = input("\nReturn to main menu? (y/n): ").strip().lower()
        if continue_session != 'y':
            print("Exiting program. Goodbye!")
            break
    else:
        print("Invalid mode selection. Please enter 1, 2, or 3.")
print("\nChatbot integration complete!")
def run_prediction_pipeline(data_path='data_set.csv', retrain=False):
    df = load_data(data_path)
    if retrain:
        X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features = preprocess_data(df)
        trained_models, cv_results = train_base_models(X_train, y_train, preprocessor)
        evaluation = evaluate_models(trained_models, X_test, y_test)
        plot_feature_importance(trained_models, preprocessor, numerical_features, categorical_features)
        stacking_model, stacking_evaluation = build_stacking_model(
            trained_models, X_train, y_train, X_test, y_test
        )
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
        pipeline = {
            'base_models': trained_models,
            'stacking_model': stacking_model,
            'preprocessor': preprocessor,
            'numerical_features': numerical_features,
            'categorical_features': categorical_features
        }
        predictions_df = make_predictions(df, pipeline, return_all_predictions=True)
    else:
        pipeline = load_pipeline()
        predictions_df = make_predictions(df, pipeline, return_all_predictions=True)
    integrate_with_chatbot(predictions_df)
    return predictions_df


import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from pydub import AudioSegment
import threading
import requests
import pyttsx3
import google.generativeai as genai

def run_audio_chatbot_v2(start_trigger='1', stop_trigger='2'):
    fs = 44100
    file_path = "user_recording.wav"
    mp3_path = file_path.replace(".wav", ".mp3")
    print(f"Press {start_trigger} to start recording...")
    start_input = input().strip()
    if start_input != start_trigger:
        print("Invalid start input. Exiting.")
        return
    recording_flag = True
    audio_frames = []
    def handle_stop():
        nonlocal recording_flag
        print(f"\nPress {stop_trigger} to stop recording...")
        while True:
            user_input = input().strip()
            if user_input == stop_trigger:
                recording_flag = False
                break
    stop_thread = threading.Thread(target=handle_stop)
    stop_thread.start()
    try:
        with sd.InputStream(samplerate=fs, channels=1, callback=lambda indata, frames, time, status: audio_frames.append(indata.copy())):
            while recording_flag:
                sd.sleep(1)
    except Exception as e:
        print(f"Error during recording: {e}")
        return
    stop_thread.join()
    if not audio_frames:
        print("No audio recorded.")
        return
    audio_array = np.concatenate(audio_frames, axis=0)
    write(file_path, fs, audio_array)
    audio = AudioSegment.from_wav(file_path)
    audio.export(mp3_path, format="mp3")
    tr_api_key = "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" 
    transcribe_url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {tr_api_key}"}
    with open(mp3_path, "rb") as file:
        files = {"file": file}
        data = {"model": "whisper-large-v3", "language": "en"}
        response = requests.post(transcribe_url, headers=headers, files=files, data=data)
    transcription = response.json()
    if 'text' not in transcription:
        print("Transcription failed:", transcription.get("error", transcription))
        return
    print("\nTranscription:\n", transcription["text"])
    chat_api_key = "AIzaSyD6VxooY8ZAop729dNk0NRFTHxBX380Lr8" 
    client = genai.Client(api_key=chat_api_key)
    try:
        gemini_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=transcription["text"]
        )
        response_text = gemini_response.text if hasattr(gemini_response, "text") else str(gemini_response)
    except Exception as e:
        print("Chatbot response error:", e)
        return
    print("\nChatbot response:\n", response_text)
    engine = pyttsx3.init()
    try:
        engine.endLoop()
    except:
        pass
    engine.say(response_text)
    engine.runAndWait()
def integrate_with_chatbot(predictions_df):
    global df_predictions, knowledge_base
    df_predictions = predictions_df
    knowledge_base = create_knowledge_base(df)
if __name__ == "__main__":
    check_and_install_requirements()
    try:
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        print("\nWelcome to the Product Price Forecasting System!")
        print("This tool helps analyze and predict product prices using machine learning.")
        running = True
        while running:
            print("\nAvailable options:")
            print("1. Train/load models and start interaction")
            print("2. Process new data for prediction")
            print("3. Exit")
            choice = input("\nSelect option (1-3): ").strip()
            if choice == "1":
                data_path = input("Enter data file path (leave blank for default 'data_set.csv'): ").strip()
                if not data_path:
                    data_path = 'data_set.csv'
                retrain = input("Retrain models? (y/n, default: n): ").strip().lower() == 'y'
                predictions_df = run_prediction_pipeline(data_path, retrain)
                print("\nAvailable interaction modes:")
                print("1. Text-to-Text Chat")
                print("2. Speech-to-Speech Chat")
                mode = input("\nSelect interaction mode (1 or 2): ")
                if mode == "1":
                    run_text_chatbot_v2(response_generator=prediction_chatbot_handler)
                elif mode == "2":
                    run_audio_chatbot_v2(start_trigger='1', stop_trigger='2', 
                                    response_generator=prediction_chatbot_handler)
            elif choice == "2":
                process_new_data()
            elif choice == "3":
                print("Exiting program. Goodbye!")
                running = False
                break
            else:
                print("Invalid option. Please try again.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure all requirements are installed correctly.")