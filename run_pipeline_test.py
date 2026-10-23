import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pywt
import scipy.ndimage as ndi
from tqdm.auto import tqdm
from PIL import Image
import timm

print("="*65)
print("   OVARIAN HISTOPATHOLOGY SWIN-CONVNEXT PIPELINE VERIFICATION   ")
print("="*65)

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Execution Device: {DEVICE}")

OUTPUT_DIR = "./pipeline_verification_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. WAVELET + BEMD-INSPIRED ENHANCEMENT
def cv2_box_filter(image, kernel_size=9):
    return ndi.uniform_filter(image, size=kernel_size, mode="reflect")

def wavelet_bemd_inspired_enhancement(pil_img):
    img = np.asarray(pil_img.convert("RGB")).astype(np.float32) / 255.0
    enhanced_channels = []
    for c in range(3):
        channel = img[:, :, c]
        cA, (cH, cV, cD) = pywt.dwt2(channel, "db1")
        reconstructed = pywt.idwt2((cA, (cH, cV, cD)), "db1")
        reconstructed = reconstructed[:channel.shape[0], :channel.shape[1]]
        local_mean = cv2_box_filter(reconstructed, kernel_size=9)
        imf = reconstructed - local_mean
        enhanced = np.clip(local_mean + 1.20 * imf, 0, 1)
        enhanced_channels.append(enhanced)
    enhanced = np.stack(enhanced_channels, axis=-1)
    return Image.fromarray((enhanced * 255).astype(np.uint8))

print("[+] Step 1: Testing Wavelet BEMD-inspired Enhancement...")
dummy_pil = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
enhanced_pil = wavelet_bemd_inspired_enhancement(dummy_pil)
print(f"    Enhancement successful! Input shape: {dummy_pil.size} -> Enhanced: {enhanced_pil.size}")

# 2. DUMMY HISTOPATHOLOGY DATASET FOR VERIFICATION
class DummyOvarianDataset(Dataset):
    def __init__(self, count=40):
        self.count = count
    def __len__(self):
        return self.count
    def __getitem__(self, idx):
        rgb_tensor = torch.randn(3, 224, 224)
        enhanced_tensor = torch.randn(3, 224, 224)
        label = idx % 5
        return rgb_tensor, enhanced_tensor, torch.tensor(label, dtype=torch.long), f"dummy_slide_{idx}.png"

train_loader = DataLoader(DummyOvarianDataset(32), batch_size=4, shuffle=True)
val_loader = DataLoader(DummyOvarianDataset(16), batch_size=4, shuffle=False)
print("[+] Step 2: Dataloaders initialized successfully.")

# 3. CROSS ATTENTION & MODEL DEFINITION
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    def forward(self, swin_tokens, conv_tokens):
        # Adaptive pooling to match token length if needed
        if swin_tokens.size(1) != conv_tokens.size(1):
            target_len = min(swin_tokens.size(1), conv_tokens.size(1))
            swin_tokens = nn.functional.adaptive_avg_pool1d(swin_tokens.transpose(1, 2), target_len).transpose(1, 2)
            conv_tokens = nn.functional.adaptive_avg_pool1d(conv_tokens.transpose(1, 2), target_len).transpose(1, 2)
        fused, _ = self.attn(query=swin_tokens, key=conv_tokens, value=conv_tokens)
        return self.norm(swin_tokens + fused)

class AttentionPooling(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.score = nn.Linear(dim, 1)
    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        pooled = torch.sum(weights * x, dim=1)
        return pooled, weights

class WaveletSwinConvNeXtTest(nn.Module):
    def __init__(self, feature_dim=256, num_classes=5):
        super().__init__()
        # Use lightweight models for instant local CPU verification
        print("    Loading test backbones (Swin-Tiny + ConvNeXt-Tiny for local verification)...")
        self.swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
        self.convnext = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
        
        self.swin_proj = nn.Linear(self.swin.num_features, feature_dim)
        self.conv_proj = nn.Linear(self.convnext.num_features, feature_dim)
        self.cross_attention = CrossAttentionFusion(dim=feature_dim, num_heads=4)
        self.attention_pool = AttentionPooling(dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb, enhanced):
        is_swin_grad = any(p.requires_grad for p in self.swin.parameters()) and self.training
        is_conv_grad = any(p.requires_grad for p in self.convnext.parameters()) and self.training
        
        with torch.set_grad_enabled(is_swin_grad):
            swin_feats = self.swin.forward_features(rgb)
            if swin_feats.ndim == 4:
                B, H, W, C = swin_feats.shape
                swin_tokens = swin_feats.reshape(B, H * W, C)
            else:
                swin_tokens = swin_feats
            swin_tokens = self.swin_proj(swin_tokens)

        with torch.set_grad_enabled(is_conv_grad):
            conv_feats = self.convnext.forward_features(enhanced)
            if conv_feats.ndim == 4:
                conv_tokens = conv_feats.flatten(2).transpose(1, 2)
            else:
                conv_tokens = conv_feats
            conv_tokens = self.conv_proj(conv_tokens)

        fused = self.cross_attention(swin_tokens, conv_tokens)
        pooled, _ = self.attention_pool(fused)
        return self.classifier(pooled)

print("[+] Step 3: Constructing Model Architecture...")
model = WaveletSwinConvNeXtTest().to(DEVICE)
print("    Model initialized successfully!")

# 4. STAGE 1 (WARMUP) EXECUTION TEST
print("\n" + "="*40)
print("     STAGE 1: WARMUP TEST (FROZEN BACKBONES)")
print("="*40)

# Freeze backbones
for p in model.swin.parameters(): p.requires_grad = False
for p in model.convnext.parameters(): p.requires_grad = False

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
pbar = tqdm(train_loader, desc="Stage 1 Warmup", leave=True)
running_loss, correct, total = 0.0, 0, 0
for rgb, enhanced, targets, _ in pbar:
    rgb, enhanced, targets = rgb.to(DEVICE), enhanced.to(DEVICE), targets.to(DEVICE)
    optimizer.zero_grad()
    logits = model(rgb, enhanced)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item() * targets.size(0)
    correct += (logits.argmax(dim=1) == targets).sum().item()
    total += targets.size(0)
    pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.1f}%"})

print(f"[OK] Stage 1 Warmup Finished! Loss: {running_loss/total:.4f}, Accuracy: {(correct/total)*100:.1f}%")
print("    >>> PROOF: Stage 1 does NOT freeze when properly configured!")

# 5. STAGE 2 (FINE-TUNING) EXECUTION TEST
print("\n" + "="*40)
print("     STAGE 2: FINE-TUNING TEST (UNFROZEN)")
print("="*40)

for p in model.swin.parameters(): p.requires_grad = True
for p in model.convnext.parameters(): p.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

pbar = tqdm(train_loader, desc="Stage 2 Fine-Tune", leave=True)
running_loss, correct, total = 0.0, 0, 0
for rgb, enhanced, targets, _ in pbar:
    rgb, enhanced, targets = rgb.to(DEVICE), enhanced.to(DEVICE), targets.to(DEVICE)
    optimizer.zero_grad()
    logits = model(rgb, enhanced)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item() * targets.size(0)
    correct += (logits.argmax(dim=1) == targets).sum().item()
    total += targets.size(0)
    pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.1f}%"})

print(f"[OK] Stage 2 Fine-Tuning Finished! Loss: {running_loss/total:.4f}, Accuracy: {(correct/total)*100:.1f}%")

# 6. VALIDATION TEST
print("\n[+] Step 6: Testing Validation Routine...")
model.eval()
val_loss, val_correct, val_total = 0.0, 0, 0
with torch.no_grad():
    for rgb, enhanced, targets, _ in val_loader:
        rgb, enhanced, targets = rgb.to(DEVICE), enhanced.to(DEVICE), targets.to(DEVICE)
        logits = model(rgb, enhanced)
        val_loss += criterion(logits, targets).item() * targets.size(0)
        val_correct += (logits.argmax(dim=1) == targets).sum().item()
        val_total += targets.size(0)
print(f"[OK] Validation Finished! Loss: {val_loss/val_total:.4f}, Acc: {(val_correct/val_total)*100:.1f}%")

# 7. SAVING VERIFICATION CHECKPOINT
save_path = os.path.join(OUTPUT_DIR, "test_ovarian_checkpoint.pth")
torch.save(model.state_dict(), save_path)
print(f"[OK] Checkpoint safely written to: {save_path}")

print("\n" + "="*65)
print("       ALL PIPELINE STAGES PASSED 100% SUCCESSFULLY!        ")
print("="*65)
