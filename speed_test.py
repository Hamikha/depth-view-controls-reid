# Speed test for the re-ID study. Needs no dataset and takes about a minute.
import time, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet50

assert torch.cuda.is_available(), "Turn on a GPU first: Runtime > Change runtime type"
device = torch.device("cuda")
net = resnet50(weights=None)
net.layer4[0].conv2.stride = (1, 1)
net.layer4[0].downsample[0].stride = (1, 1)
net.fc = nn.Identity()
neck, head = nn.BatchNorm1d(2048), nn.Linear(2048, 89, bias=False)
model = nn.ModuleList([net, neck, head]).to(device).train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-4, fused=True)
scaler = torch.amp.GradScaler()
images = torch.randint(0, 256, (356, 3, 256, 128), dtype=torch.uint8, device=device)
labels = torch.arange(12, device=device).repeat_interleave(4)

def augment(x):
    x, n = x.float().div(255).clone(), len(x)
    flip = torch.rand(n, device=device) < 0.5
    x[flip] = x[flip].flip(-1)
    theta = torch.eye(2, 3, device=device).repeat(n, 1, 1)
    theta[:, :2, 2] = (torch.rand(n, 2, device=device) * 2 - 1) * 0.1
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    gain = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, device=device)
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = ((x - mean) * gain + mean).clamp(0, 1)
    target = 256 * 128 * (0.02 + 0.31 * torch.rand(n, device=device))
    ratio = 0.3 + 3.0 * torch.rand(n, device=device)
    h, w = (target * ratio).sqrt().round().clamp(1, 256), (target / ratio).sqrt().round().clamp(1, 128)
    top = (torch.rand(n, device=device) * (256 - h + 1)).floor().view(-1, 1, 1)
    left = (torch.rand(n, device=device) * (128 - w + 1)).floor().view(-1, 1, 1)
    rows, cols = torch.arange(256, device=device).view(1, 256, 1), torch.arange(128, device=device).view(1, 1, 128)
    inside = (rows >= top) & (rows < top + h.view(-1, 1, 1)) & (cols >= left) & (cols < left + w.view(-1, 1, 1))
    selected = (torch.rand(n, device=device) < 0.5).view(-1, 1, 1)
    return torch.where((inside & selected).unsqueeze(1), torch.rand(n, 3, 1, 1, device=device), x)

def step():
    batch = augment(images[torch.randint(0, 356, (48,), device=device)])
    with torch.autocast("cuda"):
        features = model[0](batch)
        logits = model[2](model[1](features))
        unit = F.normalize(features, dim=1)
        distance = torch.cdist(unit, unit)
        same = labels[None] == labels[:, None]
        triplet = F.relu((distance * same).max(1).values - (distance + same * 1e6).min(1).values + 0.3).mean()
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1) + triplet
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

for _ in range(30):
    step()
torch.cuda.synchronize()
started = time.perf_counter()
for _ in range(150):
    step()
torch.cuda.synchronize()
speed = 150 / (time.perf_counter() - started)
minutes = 1500 / speed / 60 + 0.3  # 0.3 min covers loading and evaluation
print(torch.cuda.get_device_name(0))
print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
print(f"Speed: {speed:.1f} iterations per second")
print(f"One model: {minutes:.1f} minutes")
print(f"Full study, 880 models one at a time: {880 * minutes / 60:.1f} hours")
