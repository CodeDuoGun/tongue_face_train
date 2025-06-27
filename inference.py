from PIL import Image
import torch
# 加载模型
model.load_state_dict(torch.load("resnet50_tongue_face_other.pth"))
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])  # ImageNet std
])

train_data = datasets.ImageFolder("dataset/", transform=transform)
def predict_image(img_path):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    class_names = train_data.classes
    return class_names[predicted.item()]

# 例子
print(predict_image("test.jpg"))
