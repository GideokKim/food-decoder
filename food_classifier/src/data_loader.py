import tensorflow as tf
import tensorflow_datasets as tfds

def prepare_food101(num_classes=10, images_per_class=100):
    # 선택할 클래스 (인기있는 음식 10개)
    selected_classes = [
        'pizza', 'sushi', 'hamburger', 'ramen', 'ice_cream',
        'french_fries', 'chocolate_cake', 'salad', 'steak', 'pasta'
    ][:num_classes]
    
    builder = tfds.builder('food101')
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            extract_dir='food101_subset',
            manual_dir='food101_subset',
            download_mode=tfds.download.GenerateMode.REUSE_DATASET_IF_EXISTS,
        )
    )
    
    # 각 클래스별로 제한된 수의 이미지만 로드
    train_ds = []
    test_ds = []
    
    for class_name in selected_classes:
        # 학습 데이터
        class_train = (builder.as_dataset(split='train')
                      .filter(lambda x: x['label'] == class_name)
                      .take(images_per_class))
        train_ds.append(class_train)
        
        # 테스트 데이터
        class_test = (builder.as_dataset(split='validation')
                     .filter(lambda x: x['label'] == class_name)
                     .take(images_per_class // 4))
        test_ds.append(class_test)
    
    # 데이터셋 합치기
    train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
    test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    
    # 이미지 전처리
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [128, 128])
        image = image / 255.0
        return image, label
    
    BATCH_SIZE = 16
    
    train_ds = (train_ds
                .map(preprocess)
                .shuffle(1000)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))
    
    test_ds = (test_ds
               .map(preprocess)
               .batch(BATCH_SIZE)
               .prefetch(tf.data.AUTOTUNE))
    
    return train_ds, test_ds, builder.info