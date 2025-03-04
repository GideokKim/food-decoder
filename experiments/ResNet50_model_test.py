# ResNet50_model_test.py

# 필요한 라이브러리 설치
# !pip install torch torchvision matplotlib numpy

# 데이터셋 준비
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 데이터셋 로드
train_dataset = datasets.ImageFolder(root='sample_data/train', transform=transform)
test_dataset = datasets.ImageFolder(root='sample_data/test', transform=transform)

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ResNet50 모델 로드
import torch
import torchvision.models as models

# 사전 훈련된 ResNet50 모델 로드
model = models.resnet50(pretrained=True)
model.eval()  # 평가 모드로 설정

# 모델 평가
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'테스트 정확도: {accuracy:.2f}%') 