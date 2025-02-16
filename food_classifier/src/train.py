import tensorflow as tf
from .data_loader import prepare_food11
from .model import create_model

def train_model(epochs=10):
    # 데이터 준비
    train_ds, val_ds, class_names = prepare_food11()
    
    # 모델 생성
    model = create_model()
    
    # 콜백 설정
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
    ]
    
    # 모델 학습
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return model, history, class_names