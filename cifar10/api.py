import io
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from fastapi import FastAPI,File,UploadFile,HTTPException
from PIL import Image

from model import ResCNN, CLASS_NAMES , mean , std

app = FastAPI(title="CIFAR-10 Classifier API")

model = ResCNN()
model.load_state_dict(torch.load("best_resnet.pth",map_location="cpu"))
model.eval()

transform = T.Compose([T.Resize((32,32)), T.ToTensor(),T.Normalize(mean,std)])

@app.get("/health")
def health():
    return{"status":"ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(x),dim=1)[0]

    top = torch.topk(probs,3)
    return{
        "predictions": [
            {"class": CLASS_NAMES[i], "confidence": round(p.item(), 4)}
            for p, i in zip(top.values, top.indices)
        ]
    }