import pandas as pd
import os
import logging
import yaml

logger = logging.getLogger('feature_selection')
logger.setLevel('DEBUG')

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_selection.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def data_selection(data_model_imputed: pd.DataFrame, target_variable: str, TOP_K = 30, MIN_ABS_CORR = 0.35, CORR_METHOD = "pearson") -> list:
    """Select features based on correlation with the target variable"""
    corr_series = (
        data_model_imputed
        .corr(method=CORR_METHOD)[target_variable]
        .drop(labels=[target_variable])
        .dropna()
    )

    corr_sorted = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)

    logger.debug("correlation matrix has been calculated using method")

    if MIN_ABS_CORR is not None:
        selected_features = corr_sorted[corr_sorted.abs() >= MIN_ABS_CORR].index.tolist()
        logger.debug(f"Selected with some minimum absolute correlation threshold has been applied")
    else:
        selected_features = corr_sorted.head(TOP_K).index.tolist()
        logger.debug("values are selected without significant correlation threshold")

    selected_features = [c for c in selected_features if c in data_model_imputed.columns]
    return selected_features

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        processed_data_path = os.path.join(data_path, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(processed_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(processed_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', processed_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        target_variable = "Mortality rate, adult, female (per 1,000 female adults)"
        selected_features = data_selection(train_data, target_variable)
        logger.info(f"Selected {len(selected_features)} features from training data")
        train_processed = train_data[selected_features].copy()
        test_processed = test_data[selected_features].copy()
        logger.info(f"Final processed features count: {len(selected_features)}")
        logger.info(f"Train data shape: {train_processed.shape}, Test data shape: {test_processed.shape}")
        train_processed[target_variable] = train_data[target_variable]
        test_processed[target_variable] = test_data[target_variable]
        save_data(train_processed, test_processed, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")
if __name__ == '__main__':
    main()