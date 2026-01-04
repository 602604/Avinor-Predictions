from flight_concurrency_predictor import TrainConfig, train, model_predict, create_XGBoost

def main():
    data_path = "data/historical_flights.csv"

    config = TrainConfig(flight_csv = data_path,
                         model_function = create_XGBoost)
    model = train(config)

    model_predict("data/schedule_oct2025_updated.csv", model)
    

if __name__ == "__main__":
    main()