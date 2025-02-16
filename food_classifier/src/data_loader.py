import tensorflow as tf
import os
import pathlib
import zipfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

# GPU 사용 가능 여부 확인 및 설정
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리 증가를 허용
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'Available GPUs: {len(gpus)}')
        print(f'GPU Devices: {gpus}')
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU")

def download_food11():
    """Food-11 데이터셋 Kaggle API를 통해 다운로드"""
    print("Starting Food-11 dataset download...")
    
    try:
        # Kaggle API 인증
        api = KaggleApi()
        api.authenticate()
        
        data_dir = pathlib.Path('food11_dataset')
        if data_dir.exists():
            shutil.rmtree(data_dir)  # 기존 디렉토리 삭제
        data_dir.mkdir(parents=True)
        
        # Kaggle 데이터셋 다운로드
        print("Downloading dataset from Kaggle...")
        api.dataset_download_files(
            'trolukovich/food11-image-dataset',
            path=data_dir,
            quiet=False
        )
        
        # 압축 파일 경로
        zip_path = data_dir / "food11-image-dataset.zip"
        
        print("Extracting dataset...")
        # 압축 해제
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # 압축파일 삭제
        zip_path.unlink()
        print("Successfully downloaded and extracted dataset")
        
        # 디렉토리 구조 정리
        source_dir = data_dir / "food11"
        for split in ['training', 'validation']:
            if (source_dir / split).exists():
                shutil.move(str(source_dir / split), str(data_dir / split))
        
        # 임시 디렉토리 삭제
        if source_dir.exists():
            shutil.rmtree(source_dir)
            
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        if data_dir.exists():
            shutil.rmtree(data_dir)
        raise

def prepare_food11(img_height=96, img_width=96, batch_size=16):
    """Food-11 데이터셋 준비"""
    if not pathlib.Path('food11_dataset/training').exists():
        download_food11()
    
    print("Loading Food-11 dataset...")
    
    # 데이터셋 로드
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'food11_dataset/training',
        validation_split=None,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        'food11_dataset/validation',
        validation_split=None,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    # 클래스 이름을 먼저 저장
    class_names = train_ds.class_names
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # 메모리 효율을 위한 옵션 설정
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    
    # 단순 정규화만 적용
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (train_ds
        .map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
             num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
        .with_options(options))
    
    val_ds = (val_ds
        .map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
             num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
        .with_options(options))
    
    return train_ds, val_ds, class_names

def preprocess_image(image):
    """이미지 전처리"""
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image