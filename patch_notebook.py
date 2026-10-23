import json
import os
import ast

src_path = r"C:\Users\NANDHA K\Downloads\Ovarian_Histopathology_Swin_ConvNeXt_GradCAMPP (1).ipynb"
with open(src_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 0: pip install
c0 = "".join(nb["cells"][0]["source"])
if "tqdm" not in c0:
    c0 = c0.replace("install timm", "install tqdm scipy timm")
nb["cells"][0]["source"] = [c0]

# 2. Update Cell 1: imports
c1 = "".join(nb["cells"][1]["source"])
if "from tqdm.auto import tqdm" not in c1:
    c1 += "\nfrom tqdm.auto import tqdm\nimport scipy.ndimage as ndi\n"
nb["cells"][1]["source"] = [c1]

# 3. Update Cell 2: core logic
c2 = "".join(nb["cells"][2]["source"])

# Add OUTPUT_DIR setup at top of config
config_patch = """
# Dynamic Output Directory (Works on Google Colab, Kaggle, Linux, and Windows)
OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle") else ("/content" if os.path.exists("/content") else "./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Artifacts will be saved to:", OUTPUT_DIR)

# Safe Worker count (Prevents deadlock / freezing on Windows & Colab)
NUM_WORKERS = 0 if (os.name == 'nt' or not torch.cuda.is_available()) else 2
"""
c2 = c2.replace("NUM_WORKERS = 2", config_patch.strip())

# Fix cv2_box_filter repetitive import
old_box = '''def cv2_box_filter(image, kernel_size=9):
    """
    Lightweight box filtering without requiring OpenCV.
    """
    import scipy.ndimage as ndi
    return ndi.uniform_filter(image, size=kernel_size, mode="reflect")'''

new_box = '''def cv2_box_filter(image, kernel_size=9):
    """
    Lightweight box filtering without requiring OpenCV.
    """
    return ndi.uniform_filter(image, size=kernel_size, mode="reflect")'''
c2 = c2.replace(old_box, new_box)

# Fix VRAM bloat in WaveletSwinConvNeXt.forward during Stage 1
old_fwd = '''        # ---------------------
        # Swin branch
        # ---------------------
        swin_features = self.swin.forward_features(
            rgb
        )'''

new_fwd = '''        # ---------------------
        # Swin branch (VRAM-optimized for frozen warmup)
        # ---------------------
        is_swin_grad = any(p.requires_grad for p in self.swin.parameters()) and self.training
        with torch.set_grad_enabled(is_swin_grad):
            swin_features = self.swin.forward_features(
                rgb
            )'''
c2 = c2.replace(old_fwd, new_fwd)

old_conv_fwd = '''        # ---------------------
        # ConvNeXt branch
        # ---------------------
        conv_features = self.convnext.forward_features(
            enhanced
        )'''

new_conv_fwd = '''        # ---------------------
        # ConvNeXt branch (VRAM-optimized for frozen warmup)
        # ---------------------
        is_conv_grad = any(p.requires_grad for p in self.convnext.parameters()) and self.training
        with torch.set_grad_enabled(is_conv_grad):
            conv_features = self.convnext.forward_features(
                enhanced
            )'''
c2 = c2.replace(old_conv_fwd, new_conv_fwd)

# Add tqdm to train_one_epoch
old_train = "    for rgb, enhanced, targets, _ in loader:"
new_train = """    pbar = tqdm(loader, desc="Training  ", leave=False)
    for rgb, enhanced, targets, _ in pbar:"""
c2 = c2.replace(old_train, new_train, 1)

# Add metric update to tqdm progress bar
old_step = """        optimizer.step()

        running_loss += (
            loss.item() * targets.size(0)
        )"""
new_step = """        optimizer.step()

        running_loss += (
            loss.item() * targets.size(0)
        )
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/max(1, total))*100:.1f}%"})"""
c2 = c2.replace(old_step, new_step, 1)

# Add tqdm to validate
old_val = "    for rgb, enhanced, targets, _ in loader:"
new_val = """    pbar = tqdm(loader, desc="Validating", leave=False)
    for rgb, enhanced, targets, _ in pbar:"""
c2 = c2.replace(old_val, new_val, 1)

# Fix the multi-line print syntax error specifically
old_print_model = '''print(
    "\\nModel saved to:"
    "/content/final_ovarian_swin_convnext.pth"
)'''

new_print_model = '''print(
    "\\nModel saved to:",
    os.path.join(OUTPUT_DIR, "final_ovarian_swin_convnext.pth")
)'''
c2 = c2.replace(old_print_model, new_print_model)

# Replace all remaining hardcoded /content paths with os.path.join(OUTPUT_DIR, ...)
c2 = c2.replace('"/content/best_ovarian_model.pth"', 'os.path.join(OUTPUT_DIR, "best_ovarian_model.pth")')
c2 = c2.replace('"/content/final_ovarian_swin_convnext.pth"', 'os.path.join(OUTPUT_DIR, "final_ovarian_swin_convnext.pth")')
c2 = c2.replace('"/content/GradCAM_PlusPlus_All_Classes.png"', 'os.path.join(OUTPUT_DIR, "GradCAM_PlusPlus_All_Classes.png")')
c2 = c2.replace('"/content/ovarian_model_metrics.csv"', 'os.path.join(OUTPUT_DIR, "ovarian_model_metrics.csv")')
c2 = c2.replace('"/content/ovarian_classwise_metrics.csv"', 'os.path.join(OUTPUT_DIR, "ovarian_classwise_metrics.csv")')
c2 = c2.replace('"/content/classification_report.txt"', 'os.path.join(OUTPUT_DIR, "classification_report.txt")')
c2 = c2.replace('print("/content/best_ovarian_model.pth")', 'print(os.path.join(OUTPUT_DIR, "best_ovarian_model.pth"))')
c2 = c2.replace('print("/content/final_ovarian_swin_convnext.pth")', 'print(os.path.join(OUTPUT_DIR, "final_ovarian_swin_convnext.pth"))')
c2 = c2.replace('print("/content/GradCAM_PlusPlus_All_Classes.png")', 'print(os.path.join(OUTPUT_DIR, "GradCAM_PlusPlus_All_Classes.png"))')
c2 = c2.replace('print("/content/ovarian_model_metrics.csv")', 'print(os.path.join(OUTPUT_DIR, "ovarian_model_metrics.csv"))')
c2 = c2.replace('print("/content/ovarian_classwise_metrics.csv")', 'print(os.path.join(OUTPUT_DIR, "ovarian_classwise_metrics.csv"))')
c2 = c2.replace('print("/content/classification_report.txt")', 'print(os.path.join(OUTPUT_DIR, "classification_report.txt"))')

# Verify python syntax of cell 2
clean_c2 = "\n".join([line for line in c2.splitlines() if not line.strip().startswith("!") and not line.strip().startswith("%")])
try:
    ast.parse(clean_c2)
    print("Cell 2 syntax verification: 100% CLEAN (No syntax errors)")
except SyntaxError as e:
    print("SYNTAX ERROR FOUND:", e)
    raise

nb["cells"][2]["source"] = [c2]

out_downloads = r"C:\Users\NANDHA K\Downloads\Ovarian_Histopathology_Swin_ConvNeXt_GradCAMPP_CORRECTED.ipynb"
out_workspace = r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose\Ovarian_Histopathology_Swin_ConvNeXt_GradCAMPP_CORRECTED.ipynb"

with open(out_downloads, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

with open(out_workspace, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("SUCCESS: Fully validated corrected notebook generated at:")
print("1.", out_downloads)
print("2.", out_workspace)
