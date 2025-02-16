import tensorflow as tf
import tensorflow_datasets as tfds

def prepare_food101():
    # 데이터셋 다운로드 및 로드
    (train_ds, test_ds), info = tfds.load(
        'food101',
        split=['train', 'validation'],
        with_info=True,
        as_supervised=True,
    )
    
    # 이미지 전처리 함수
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [224, 224])
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image, label
    
    # 데이터셋 설정
    AUTOTUNE = tf.data.AUTOTUNE
    BATCH_SIZE = 32
    
    train_ds = (train_ds
                .map(preprocess, num_parallel_calls=AUTOTUNE)
                .shuffle(1000)
                .batch(BATCH_SIZE)
                .prefetch(AUTOTUNE))
    
    test_ds = (test_ds
               .map(preprocess, num_parallel_calls=AUTOTUNE)
               .batch(BATCH_SIZE)
               .prefetch(AUTOTUNE))
    
    return train_ds, test_ds, info