import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import AutoModel, AutoConfig

BACKBONE_ID = "OpenGVLab/VideoMAEv2-Base"

# Names of attention projection layers to inject LoRA into.
# VideoMAEv2 uses a fused 'qkv' projection; separate q/v names are listed as
# fallback for other ViT variants.
_LORA_TARGET_NAMES = {'qkv', 'q', 'v', 'q_proj', 'v_proj'}


def load_backbone() -> nn.Module:
    config   = AutoConfig.from_pretrained(BACKBONE_ID, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(BACKBONE_ID, config=config, trust_remote_code=True)
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def inject_lora(backbone: nn.Module, rank: int) -> int:
    """Replace matching attention projections in backbone with LoRALinear.

    Walks named_modules, finds nn.Linear layers whose leaf name is in
    _LORA_TARGET_NAMES, and swaps them in-place. Returns the count of
    injected layers so the caller can verify something was found.
    """
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []

    for full_name, module in backbone.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = full_name.split('.')[-1]
        if leaf not in _LORA_TARGET_NAMES:
            continue
        parent = backbone
        for part in full_name.split('.')[:-1]:
            parent = getattr(parent, part)
        replacements.append((parent, leaf, module))

    for parent, leaf, original in replacements:
        setattr(parent, leaf, LoRALinear(original, rank))

    return len(replacements)


class VideoClassifier(nn.Module):
    def __init__(self, head: nn.Module, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)  →  model expects (B, C, T, H, W)
        pixel_values = x.permute(0, 2, 1, 3, 4)
        outputs   = self.backbone(pixel_values=pixel_values)
        # VideoMAEv2 returns a plain tensor; other HF models return ModelOutput
        hidden    = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs
        cls_token = hidden   # (B, 768) — backbone already returns pooled representation
        return self.head(cls_token)


def build_model(head_type: str, num_classes: int, lora_rank: int = 16) -> VideoClassifier:
    backbone = load_backbone()

    if head_type == 'linear':
        head = LinearHead(num_classes)

    elif head_type == 'mlp':
        head = MLPHead(num_classes)

    elif head_type == 'lora':
        n = inject_lora(backbone, lora_rank)
        print(f"[inject_lora] replaced {n} layer(s) with LoRALinear (rank={lora_rank})")
        if n == 0:
            raise RuntimeError(
                "inject_lora found no target layers. "
                "Inspect backbone with: "
                "  for n, m in backbone.named_modules(): print(n, type(m))\n"
                "then add the correct leaf name to _LORA_TARGET_NAMES in model.py."
            )
        head = LoRAHead(num_classes)

    else:
        raise ValueError(f"Unknown head type: {head_type!r}")

    return VideoClassifier(head=head, backbone=backbone)


# ─────────────────────────────────────────────────────────────────────────────
# Implement the four classes below yourself.
# Each class has a docstring that describes exactly what it should do.
# Run `python train.py --head linear ...` to test as you go.
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a LoRA branch.

    Forward pass:
        output = original_linear(x) + lora_B( lora_A(x) ) / rank

    What to put in __init__:
        self.original_linear  — store the frozen linear passed in
        self.rank             — already set below, keep it
        self.lora_A           — nn.Linear(in_features, rank, bias=False)
                                initialise with nn.init.kaiming_uniform_
        self.lora_B           — nn.Linear(rank, out_features, bias=False)
                                initialise with nn.init.zeros_
        (initialising B to zero means LoRA adds nothing at the start of training)
    """
    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        self.rank = rank
        # TODO ↓
        self.original_linear    = linear
        self.rank = rank

        self.lora_A = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, linear.out_features)

        init.kaiming_uniform_(self.lora_A.weight)
        init.zeros_(self.lora_B.weight)
    
    @property
    def weight(self):
        return self.original_linear.weight + (self.lora_B.weight @ self.lora_A.weight) / self.rank
    
    @property
    def bias(self):
        return self.original_linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO ↓
        original_x  = self.original_linear(x)
        lora_x      = self.lora_B(self.lora_A(x)) / self.rank

        return original_x + lora_x
        # raise NotImplementedError


class LinearHead(nn.Module):
    """Single linear layer: Linear(768 → num_classes)."""
    def __init__(self, num_classes: int):
        super().__init__()
        # TODO ↓
        self.linear = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 768)
        # TODO ↓
        return self.linear(x)
        # raise NotImplementedError


class MLPHead(nn.Module):
    """MLP classifier: Linear(768→512) → GELU → Dropout(0.3) → Linear(512→num_classes)."""
    def __init__(self, num_classes: int):
        super().__init__()
        # TODO ↓
        self.linear     = nn.Linear(768, 512)
        self.gelu       = nn.GELU()
        self.drop       = nn.Dropout(0.3)
        self.linear2    = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 768)
        # TODO ↓
        x = self.linear(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x
        # raise NotImplementedError


class LoRAHead(nn.Module):
    """Classification head used alongside LoRA-injected backbone.

    The LoRA adapters live inside the backbone (injected by inject_lora).
    This head just maps the CLS token to class logits: Linear(768 → num_classes).
    """
    def __init__(self, num_classes: int):
        super().__init__()
        # TODO ↓
        self.map_linear = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 768)
        # TODO ↓
        return self.map_linear(x)
        # raise NotImplementedError
