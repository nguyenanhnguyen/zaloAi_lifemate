"""Chứng minh chunked_cross_entropy cho ra cùng loss VÀ cùng gradient với bản thường.

    python scripts/diag/check_chunked_loss.py

So giá trị loss là chưa đủ. Chỗ dễ sai là backward: mỗi khối được bọc trong gradient
checkpointing nên logits bị tính lại lúc backward, và nếu việc tính lại lệch so với lúc
forward thì loss vẫn in ra đúng trong khi gradient sai. Model sẽ học sai mà mọi con số
trên log đều đẹp. Nên phép so chính ở đây là gradient theo hidden.

Kiểm thêm hai thứ dễ bỏ sót: mẫu có vị trí bị che bằng IGNORE (phần lớn chuỗi của ta là
vậy -- chỉ 1.5% token được tính loss), và khối có vị trí bị che TOÀN BỘ.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from train_sft import IGNORE, chunked_cross_entropy  # noqa: E402


def reference(head, hidden, labels):
    logits = head(hidden).float()
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        ignore_index=IGNORE, reduction="sum")


def case(name, *, length, chunk, vocab, hidden_size, supervised, dtype):
    torch.manual_seed(0)
    head = torch.nn.Linear(hidden_size, vocab, bias=False, dtype=dtype)
    base = torch.randn(1, length, hidden_size, dtype=dtype)

    labels = torch.randint(0, vocab, (1, length))
    if supervised == "thưa":
        keep = torch.rand(1, length) < 0.015       # cỡ tỉ lệ thật trong sft.train.jsonl
        labels = labels.masked_fill(~keep, IGNORE)
    elif supervised == "khối rỗng":
        labels[:, chunk:2 * chunk] = IGNORE        # một khối không có token nào được tính

    results = {}
    for which, fn in (("thường", reference),
                      ("theo khối", lambda h, l: chunked_cross_entropy(head, h, l, chunk))):
        hidden = base.clone().requires_grad_(True)
        loss = fn(head, hidden, labels) if which == "thường" else fn(hidden, labels)
        loss.backward()
        results[which] = (loss.detach().clone(), hidden.grad.clone())

    (loss_a, grad_a), (loss_b, grad_b) = results["thường"], results["theo khối"]
    loss_gap = (loss_a - loss_b).abs().item()
    grad_gap = (grad_a - grad_b).abs().max().item()
    scale = grad_a.abs().max().item()
    ok = loss_gap <= 1e-3 and grad_gap <= 1e-6
    print(f"  {'OK ' if ok else 'LỆCH'} {name:48s} loss {loss_a.item():10.4f} "
          f"lệch {loss_gap:.2e} | gradient lệch {grad_gap:.2e} (|grad| lớn nhất {scale:.2e})")
    return ok


def main():
    print(f"torch {torch.__version__}\n")
    results = []
    for supervised in ("đầy đủ", "thưa", "khối rỗng"):
        for length, chunk in ((512, 128), (500, 128), (128, 1024)):
            results.append(case(
                f"L={length:4d} khối={chunk:4d} nhãn {supervised}",
                length=length, chunk=chunk, vocab=2048, hidden_size=64,
                supervised=supervised, dtype=torch.float32))
    print(f"\n{sum(results)}/{len(results)} phép so đạt")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
