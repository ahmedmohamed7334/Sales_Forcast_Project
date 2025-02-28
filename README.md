# Sales Forecast Project

## Project Overview
The Sales Forecast Project is currently under development. All files, data structures, and functionalities are subject to change based on project requirements and improvements. The system is designed to leverage hierarchical neural networks for predictive analytics using CSV file inputs

## System Architecture
The project utilizes a multi-layered neural network approach:
- **Domain-Specific Neural Networks**
  - LSTM models for time-series data
  - Fully connected neural networks or transformer-based models for tabular data
- **Supervising Neural Network**
  - Aggregates outputs from domain-specific models
  - Refines final predictions based on global patterns

## Data Processing Pipeline
1. **Data Input:** Users upload structured CSV files.
2. **Preprocessing:** Standardization, normalization, categorical encoding, and missing value handling.
3. **Training Phase:**
   - Train individual domain-specific networks.
   - Train the supervising network to integrate outputs.
4. **Model Evaluation:**
   - Time-series models: RMSE, MAE, MASE
   - Tabular models: Accuracy, Precision, Recall
   - Supervising model: Combined accuracy and alignment metrics
5. **Prediction Pipeline:**
   - New CSV data is processed through domain-specific networks.
   - Predictions are aggregated and refined by the supervising model.

## API Integrations
The system integrates various AI-powered APIs to enhance functionality:
- **Cohere Semantic Search:** Retrieves relevant data for queries.
- **Whisper API:** Converts voice input into text.
- **Gemini 2 flash API:** Generates text-based explanations.
- **Text-to-Speech API:** Converts insights into audio.
- **Text-to-Video API:** Produces video explanations using virtual spokespersons.

## Key Considerations
- **Data Integrity:** Ensures clean and structured CSV inputs.
- **Model Flexibility:** Supports various data types and domains.
- **Continuous Improvement:** Iterative updates based on evaluation metrics.

## Disclaimer
This project is actively evolving. Features, models, and file structures are subject to updates based on research, testing, and development requirements.

