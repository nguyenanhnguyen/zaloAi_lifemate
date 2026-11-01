"""Tìm tên dùng TRƯỚC khi gán, chạy qua các cell theo đúng thứ tự thực thi.

Đây là lớp lỗi đã làm hỏng một lần chạy Kaggle: `config={... 'transcript_cache_size':len(TRANSCRIPTS)}`
nằm TRƯỚC `TRANSCRIPTS = load_transcript_cache(...)` trong cùng một cell, nên `NameError` chỉ lộ ra
sau khi đã tải 13 GiB model và warm GPU — tức sau 3,5 phút quota. Cú pháp sạch không bắt được nó.

Luật, chọn để bắt đúng lớp lỗi đó mà không ngập dương tính giả:

* Đi **tuần tự giữa** các câu lệnh mức module, cộng dồn tên đã gán.
* **Trong** một câu lệnh ghép (`for`/`while`/`if`/`with`/`try`) thì dễ dãi: coi như mọi tên nó gán ở
  bất cứ đâu bên trong đều đã có. Thân vòng lặp chạy nhiều lần và nhánh `except` thấy biến của thân
  `try`, nên siết ở đây chỉ sinh ồn.
* Thân hàm/class bị bỏ qua (phân giải lúc gọi), nhưng **default của tham số thì không** — nó tính ở
  mức module.
* Comprehension có scope riêng; walrus bind ra ngoài.

    python scripts/diag/check_notebook_order.py notebook/lifemate_kaggle_v2.ipynb
"""
from __future__ import annotations

import ast
import builtins
import json
import sys
from pathlib import Path

FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
DEFERRED = FUNCTIONS + (ast.ClassDef,)
COMPOUND = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try)
ASSIGNMENTS = (ast.Assign, ast.AnnAssign, ast.AugAssign)


def bound_names(node: ast.AST) -> set[str]:
    """Mọi tên mà node này (kể cả bên trong) gán."""
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            names.update(child.names)
    return names


def guarded_names(statement: ast.stmt) -> set[str]:
    """`if 'monitor' in globals():` là chốt bảo vệ có chủ ý — tên đó không phải lỗi thứ tự.

    Mẫu này dùng để chạy lại một cell trong session tương tác mà không rò trạng thái cũ, nên nó
    phải đọc một tên có thể chưa tồn tại. Chỉ nhận đúng dạng đó, không mute cả câu lệnh.
    """
    if not isinstance(statement, ast.If) or not isinstance(statement.test, ast.Compare):
        return set()
    test = statement.test
    if not (len(test.ops) == 1 and isinstance(test.ops[0], ast.In)
            and isinstance(test.left, ast.Constant) and isinstance(test.left.value, str)):
        return set()
    target = test.comparators[0]
    if isinstance(target, ast.Call) and isinstance(target.func, ast.Name) \
            and target.func.id in {"globals", "locals", "dir", "vars"}:
        return {test.left.value}
    return set()


def loads(node: ast.AST, known: set[str], found: list[tuple[str, int]]) -> None:
    """Thu tên được ĐỌC trong một biểu thức, tôn trọng scope của comprehension."""
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        inner = set(known)
        for generator in node.generators:
            loads(generator.iter, inner, found)
            inner |= bound_names(generator.target)
            for condition in generator.ifs:
                loads(condition, inner, found)
        for part in (getattr(node, "elt", None), getattr(node, "key", None),
                     getattr(node, "value", None)):
            if part is not None:
                loads(part, inner, found)
        return
    if isinstance(node, FUNCTIONS):
        # Thân lambda/def hoãn tới lúc gọi; default thì tính ngay.
        for default in list(node.args.defaults) + [d for d in node.args.kw_defaults if d]:
            loads(default, known, found)
        return
    if isinstance(node, ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id not in known:
            found.append((node.id, node.lineno))
        return
    if isinstance(node, ast.NamedExpr):           # walrus: bind ra scope ngoài
        loads(node.value, known, found)
        known |= bound_names(node.target)
        return
    for child in ast.iter_child_nodes(node):
        loads(child, known, found)


def scan(statements: list[ast.stmt], known: set[str]) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for statement in statements:
        if isinstance(statement, DEFERRED):
            if isinstance(statement, ast.ClassDef):
                for base in statement.bases:
                    loads(base, known, found)
            else:
                for default in (list(statement.args.defaults)
                                + [d for d in statement.args.kw_defaults if d]):
                    loads(default, known, found)
            known |= bound_names(statement)
            continue
        if isinstance(statement, COMPOUND):
            # Dễ dãi bên trong: thân chạy nhiều lần, except thấy biến của try.
            inside = known | bound_names(statement)
            inside |= guarded_names(statement)
            for child in ast.iter_child_nodes(statement):
                if isinstance(child, ast.stmt):
                    found += scan([child], inside)
                elif isinstance(child, ast.excepthandler):
                    found += scan(child.body, inside)
                else:
                    loads(child, inside, found)
            known |= bound_names(statement)
            continue
        if isinstance(statement, ASSIGNMENTS):
            # Target KHÔNG được coi là đã có: `x = x + 1` lần đầu là NameError thật.
            if statement.value is not None:
                loads(statement.value, known, found)
            if isinstance(statement, ast.AugAssign):
                loads(statement.target, known, found)
            known |= bound_names(statement)
            continue
        for child in ast.iter_child_nodes(statement):
            if isinstance(child, ast.stmt):
                found += scan([child], known)
            else:
                loads(child, known, found)
        known |= bound_names(statement)
    return found


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "notebook/lifemate_kaggle_v2.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]

    known = set(dir(builtins)) | {"__name__", "__file__", "get_ipython", "In", "Out", "_"}
    problems: list[str] = []
    for index, cell in enumerate(code):
        source = "".join(cell["source"])
        lines = source.splitlines()
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            problems.append(f"cell {index}: SyntaxError {error}")
            continue
        for name, line in scan(tree.body, set(known)):
            text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            problems.append(f"cell {index} dòng {line}: '{name}' đọc trước khi gán\n      {text[:110]}")
        known |= bound_names(tree)

    print(f"{len(code)} cell code, {len(problems)} vấn đề")
    for problem in problems:
        print("  " + problem)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
