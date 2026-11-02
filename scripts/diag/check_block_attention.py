"""Chứng minh block_attention_forward cho ra cùng kết quả với đường sdpa thường.

    python scripts/diag/check_block_attention.py

Attention theo khối là thứ duy nhất trong pipeline mà tôi tự viết lại phần toán, nên nó
phải được kiểm riêng. Một bản attention sai lệch nhẹ không làm training nổ -- nó làm model
học sai mà vẫn ra loss đẹp, đúng kiểu lỗi đắt nhất.

Phép so là với đường tham chiếu rõ ràng: repeat_kv rồi gọi SDPA một lần với is_causal=True.
Kiểm cả ba thứ dễ sai: GQA (32 query head trên 8 kv head), độ dài không chia hết cho khối,
và nhánh có attention_mask truyền vào.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

import train_sft  # noqa: E402


class FakeAttention(torch.nn.Module):
    """Chỉ cần đúng một thuộc tính: số nhóm GQA."""

    def __init__(self, groups: int) -> None:
        super().__init__()
        self.num_key_value_groups = groups


def reference(module, query, key, value, attention_mask, scaling):
    from transformers.integrations.sdpa_attention import repeat_kv
    if module.num_key_value_groups > 1:
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    out = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, scale=scaling,
        is_causal=attention_mask is None)
    return out.transpose(1, 2).contiguous()


def case(name, *, length, block, groups, masked, device, dtype):
    train_sft.QUERY_BLOCK = block
    generator = torch.Generator(device="cpu").manual_seed(0)
    heads, kv_heads, dim = 32, 32 // groups, 128
    shape = (1, heads, length, dim)
    kv_shape = (1, kv_heads, length, dim)
    query = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    key = torch.randn(kv_shape, generator=generator).to(device=device, dtype=dtype)
    value = torch.randn(kv_shape, generator=generator).to(device=device, dtype=dtype)

    mask = None
    if masked:
        # Dạng mask transformers truyền khi KHÔNG bỏ được causal mask: additive, 4 chiều,
        # 0 ở chỗ được nhìn và -inf ở chỗ bị che.
        allowed = torch.ones(length, length, dtype=torch.bool, device=device).tril()
        mask = torch.zeros(1, 1, length, length, device=device, dtype=dtype)
        mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)

    module = FakeAttention(groups).to(device)
    scaling = dim ** -0.5
    got, weights = train_sft.block_attention_forward(
        module, query, key, value, mask, scaling=scaling)
    want = reference(module, query, key, value, mask, scaling)

    assert weights is None, "phải trả None ở chỗ attn_weights, như sdpa_attention_forward"
    assert got.shape == want.shape == (1, length, heads, dim), f"hình dạng sai: {got.shape}"
    gap = (got.float() - want.float()).abs().max().item()
    scale = want.float().abs().max().item()
    limit = 2e-3 if dtype is torch.float16 else 1e-5
    status = "OK " if gap <= limit else "LỆCH"
    print(f"  {status} {name:52s} lệch lớn nhất {gap:.2e} (biên {limit:.0e}, "
          f"|want| lớn nhất {scale:.2f})")
    return gap <= limit


def main():
    print(f"torch {torch.__version__}")
    results = []

    print("\nCPU, fp32 -- phép so chính xác, bắt lỗi logic che mask")
    for groups in (1, 4):
        for length, block in ((256, 64), (300, 64), (64, 1024)):
            for masked in (False, True):
                results.append(case(
                    f"L={length:4d} khối={block:4d} nhóm GQA={groups} "
                    f"{'có mask' if masked else 'không mask'}",
                    length=length, block=block, groups=groups, masked=masked,
                    device="cpu", dtype=torch.float32))

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        print(f"\nGPU {name} (sm{major}{minor}), fp16 -- đúng dtype lúc huấn luyện")
        for length, block in ((2048, 512), (2000, 512)):
            results.append(case(f"L={length:4d} khối={block:4d} nhóm GQA=4 không mask",
                                length=length, block=block, groups=4, masked=False,
                                device="cuda", dtype=torch.float16))
    else:
        print("\nKhông thấy GPU -- bỏ phần fp16")

    print(f"\n{sum(results)}/{len(results)} phép so đạt")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
