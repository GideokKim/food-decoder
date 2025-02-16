import tensorflow as tf
import numpy as np

def predict_food(model, image_path, dataset_info):
    # 이미지 로드 및 전처리
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    # 예측
    predictions = model.predict(img_array)
    score = predictions[0]
    
    # 클래스 이름 가져오기
    class_names = dataset_info.features['label'].names
    
    # 상위 3개 예측 결과 반환
    top_3 = np.argsort(score)[-3:][::-1]
    results = [
        (class_names[i], float(score[i]) * 100) 
        for i in top_3
    ]
    
    return results
