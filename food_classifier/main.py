from src.train import train_model
from src.predict import predict_food
from utils.visualization import plot_training_history

def main():
    # 모델 학습
    model, history, dataset_info = train_model()
    
    # 학습 결과 시각화
    plot_training_history(history)
    
    # 예측 테스트
    image_path = "test_food_image.jpg"  # 테스트할 이미지 경로
    results = predict_food(model, image_path, dataset_info)
    
    for food, confidence in results:
        print(f"{food}: {confidence:.2f}%")

if __name__ == "__main__":
    main()
