import tensorflow as tf
from .data_loader import prepare_food101
from .model import create_model

def train_model():
    # 데이터 로드
    train_ds, test_ds, dataset_info = prepare_food101()
    
    # 모델 생성
    model = create_model()
    
    # 콜백 설정
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3
        )
    ]
    
    # 모델 학습
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=10,
        callbacks=callbacks
    )
    
    return model, history, dataset_info