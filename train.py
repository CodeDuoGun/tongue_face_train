import torch
import torch.nn as nn
from torchvision import models,datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

"""
dataset/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── labels.csv

filename,face,tongue
img001.jpg,1,0
img002.jpg,1,1
img003.jpg,0,1
img004.jpg,0,0

"""
# 自定义Dataset
class MultiLabelDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor([row['face'], row['tongue']], dtype=torch.float32)
        return image, labels

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# 数据加载
dataset = MultiLabelDataset('dataset/labels.csv', 'dataset/images', transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 加载 ResNet50
model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 2),
    nn.Sigmoid()  # 输出 [0-1] 概率
)

# 训练设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)           # 输出 shape [B, 2]
        loss = criterion(outputs, labels) # 多标签用 BCE Loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# 保存模型
torch.save(model.state_dict(), "tongue_face_other_model.pth")
print("Model saved as tongue_face_other_model.pth")