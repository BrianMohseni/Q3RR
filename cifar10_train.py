import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
from q3r_utils import Q3RModule, AdamQ3R

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, emb_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class ViTTBlock(nn.Module):
    def __init__(self, emb_dim=192, num_heads=3, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, int(emb_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(emb_dim * mlp_ratio), emb_dim)
        )

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + h
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h
        return x

class ViTTiny(nn.Module):
    def __init__(self, img_size=32, patch_size=4, emb_dim=192, depth=4, num_heads=3, mlp_ratio=4.0, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, emb_dim))
        self.blocks = nn.ModuleList([
            ViTTBlock(emb_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

def low_rank_copy(param, keep_ratio):
    W = param.data
    shape_orig = W.shape
    if W.ndim != 2:
        W = W.view(W.shape[0], -1)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = max(1, int(S.numel() * keep_ratio))
    k = (k // 8) * 8
    k = max(k, 8)
    S_trunc = S[:k]
    U_trunc = U[:, :k]
    Vt_trunc = Vt[:k, :]
    W_new = (U_trunc @ torch.diag(S_trunc) @ Vt_trunc).to(param.device)
    return W_new.view(shape_orig)

def validate_low_rank(model, testloader, ratios=[0.1,0.2,0.25,0.5,0.75,1.0]):
    model.eval()
    results = {}
    for r in ratios:
        state_dict = {n: p.data.clone() for n, p in model.named_parameters() if p.ndim==2}
        for n, p in model.named_parameters():
            if p.ndim ==2:
                p.data.copy_(low_rank_copy(state_dict[n], r))
        correct, total = 0,0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                _, pred = out.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
        results[r] = correct/total
        for n, p in model.named_parameters():
            if p.ndim==2:
                p.data.copy_(state_dict[n])
    return results

def main():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])
    
    torch.backends.cudnn.benchmark=True
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4) # num workers tends to have some annoyances on windows, should be fixed.

    model = ViTTiny().to(device)
    params_q3r, params_other = [], []
    for n,p in model.named_parameters():
        if p.ndim==2:
            params_q3r.append(p)
        else:
            params_other.append(p)
    q3r = Q3RModule(lambda_reg=1e-3, eps=1e-3, recompute_every=100, target_rank=16, use_truncated=True)
    optimizer = AdamQ3R([
        {"params": params_q3r, "q3r": True, "lr":3e-4},
        {"params": params_other, "lr":3e-4}
    ], q3r_module=q3r, weight_decay=1e-1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        model.train()
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
        val_results = validate_low_rank(model, testloader)
        print(f"Epoch {epoch+1}")
        for r, acc in val_results.items():
            print(f"Epoch {epoch+1} | {int(r*100)}% params -> Val Accuracy: {acc:.4f}")
        
    results = validate_low_rank(model, testloader)
    for r, acc in results.items():
        print(f"{int(r*100)}% params -> Test Accuracy: {acc:.4f}")

if __name__=="__main__":
    main()
