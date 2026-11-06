# House Price Prediction – MLOps Lab 6–7

## 1. Project Overview

This project develops a Machine Learning application that predicts house prices based on four input features:

- Area in square feet
- Number of bedrooms
- Age of the house
- Location score

The project uses a Random Forest Regression model and demonstrates an end-to-end ML workflow, including data validation, model training, evaluation, prediction, API development, automated testing, and Docker deployment.

## 2. Technologies Used

- Python 3.11
- Pandas
- Scikit-learn
- Joblib
- FastAPI
- PyYAML
- Pytest
- Docker
- Docker Compose

## 3. Project Structure

```text
MLOps-Lab-6-7/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
│
├── models/
│   └── house_price_pipeline.pkl
│
├── reports/
│   └── evaluation_report.json
│
├── src/
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── api.py
│
├── tests/
│   ├── test_api.py
│   └── test_pipeline.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 4. Machine Learning Pipeline

The project follows these steps:

1. Load the house price dataset.
2. Validate the dataset.
3. Separate input features and target variable.
4. Split the dataset into training and testing sets.
5. Train a Random Forest Regression model.
6. Evaluate the model using MAE, RMSE, and R² score.
7. Save the trained model using Joblib.
8. Generate predictions for new house details.

The dataset contains 33 records. The training set contains 26 records and the testing set contains 7 records.

## 5. Model Evaluation

The model was evaluated using the following metrics:

| Metric | Result |
|---|---:|
| MAE | 1518485.71 |
| RMSE | 3972086.71 |
| R² Score | -0.1352 |

The results indicate that the model has limited predictive performance on this small dataset. Further improvements require more representative training data and model tuning.

## 6. Running the Project Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python -m src.train
```

Evaluate the model:

```bash
python -m src.evaluate
```

Generate predictions:

```bash
python -m src.predict
```

Run automated tests:

```bash
python -m pytest -v
```

## 7. FastAPI Application

Start the API:

```bash
uvicorn src.api:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

### API Endpoints

**Health Check**

- Method: GET
- Endpoint: `/health`
- Checks whether the API is running.

**House Price Prediction**

- Method: POST
- Endpoint: `/predict`
- Accepts house details and returns the predicted price.

Example request:

```json
{
  "area_sqft": 1000,
  "bedrooms": 2,
  "age_years": 10,
  "location_score": 7
}
```

## 8. Docker Deployment

Build the Docker image:

```bash
docker build -t house-price-api .
```

Run the container:

```bash
docker run -d -p 8000:8000 --name house-price-container house-price-api
```

Check running containers:

```bash
docker ps
```

The API is available at:

```text
http://127.0.0.1:8000/docs
```

## 9. Docker Compose

The project also supports Docker Compose.

Start the application:

```bash
docker compose up --build -d
```

Check the service:

```bash
docker compose ps
```

Stop the application:

```bash
docker compose down
```

## 10. Testing

Automated tests were performed for:

- Dataset loading
- Dataset validation
- Data splitting
- Saved model availability
- API health endpoint
- Valid prediction requests
- Invalid prediction requests

**Test result: 7 tests passed successfully.**

The Docker deployment was also tested. The local and Docker prediction results matched for the same input.

## 11. Conclusion

This project demonstrates an end-to-end Machine Learning workflow using modular Python code, FastAPI, automated testing, Docker, and Docker Compose.

The application successfully trains and saves a house price prediction model, provides predictions through an API, and runs inside a Docker container. The project also demonstrates consistent predictions between local execution and Docker deployment for the tested input.