# Quick usage

```python
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

model_type = "vit_t"
sam_checkpoint = "./textsam.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

textsam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
textsam.to(device=device)
textsam.eval()

predictor = SamPredictor(textsam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```