from PIL import Image
from torchvision import transforms
import torch
# 加载模型
model.load_state_dict(torch.load("resnet50_tongue_face_other.pth"))

# 加载模型
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_image(img_path):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor).cpu().numpy()[0]
        face_score, tongue_score = output
        return {
            "face": int(face_score > 0.5),
            "tongue": int(tongue_score > 0.5),
            "raw_probs": output
        }

# 示例预测
result = predict_image('dataset/images/img001.jpg')
print(result)
