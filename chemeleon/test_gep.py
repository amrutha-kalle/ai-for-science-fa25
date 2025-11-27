# test_stoich.py
import torch
from chemeleon.text_encoder.crystal_clip import CrystalClip
from chemeleon.constants import (
    PATH_CLIP_GENERAL_TEXT,
    PATH_CHEMELEON_GENERAL_TEXT,
    PATH_CLIP_COMPOSITION,
    PATH_CHEMELEON_COMPOSITION,
    CHECKPOINT_URLS,
)
# 1) Load your CLIP checkpoint (the one you just modified / trained)
clip_ckpt = PATH_CLIP_GENERAL_TEXT   # <-- put your path here
clip = CrystalClip.load_from_checkpoint(clip_ckpt, map_location="cpu")
clip.eval()

print("clip_dim:", clip.clip_dim)
print("use_gep:", getattr(clip, "use_gep", None))

texts = [
    "LiFePO4 in the olivine structure",
    "TiO2 in rutile crystal system",
]

with torch.no_grad():
    z_cond = clip.get_conditioning_embeds(texts)   # GEP+fusion if use_gep=True
    z_base = clip.get_text_embeds(texts)           # just CLIP text_proj

print("z_cond shape:", z_cond.shape)
print("z_base shape:", z_base.shape)
print("||z_cond - z_base|| per sample:", torch.norm(z_cond - z_base, dim=-1))
