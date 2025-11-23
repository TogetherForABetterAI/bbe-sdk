import torch
import torch.nn.functional as F

# --- CNN model definition ---
class ImageClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3), torch.nn.ReLU()
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 22 * 22, 10)
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))
    
    def predict(self, image):
        img_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self(img_tensor)
            probs = F.softmax(logits, dim=1).squeeze().tolist()

        return probs
