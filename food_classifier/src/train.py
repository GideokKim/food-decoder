import tensorflow as tf
from .data_loader import prepare_food101
from .model import create_model

def train_model(num_classes=10, epochs=5):
    # 데이터 로드
    train_ds, test_ds, dataset_info = prepare_food101(num_classes=num_classes)
    
    # 모델 생성
    model = create_model(num_classes=num_classes)
    
    # 모델 학습
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs
    )
    
    return model, history, dataset_info