import torch
import torch.nn as nn
from torchvision import models,datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

"""
dataset/
├── tongue/
│   ├── img001.jpg
│   └── ...
├── face/
│   ├── img101.jpg
│   └── ...
└── other/
    ├── img201.jpg
    └── ...

"""
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])  # ImageNet std
])

train_data = datasets.ImageFolder("dataset/", transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
# 替换最后一层为三分类
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)

# 训练配置

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练
for epoch in range(10):  # 可自行调整轮数
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# 保存模型
torch.save(model.state_dict(), "tongue_face_other_model.pth")
print("Model saved as tongue_face_other_model.pth")