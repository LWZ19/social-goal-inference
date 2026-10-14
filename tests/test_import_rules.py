import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
INFERENCE = ("inputs", "geometry", "imaging", "types", "vlm", "schemas", "prompts", "stages", "pipeline", "solver")
FORBIDDEN = ("scene", "freespace", "gt", "evaluate", "check_scene", "synth", "viz", "run")


def inference_files(src: Path = SRC) -> list[Path]:
    files = []
    for name in INFERENCE:                                   # each is a module file or a package folder
        files += [p for p in [src / "sgi" / f"{name}.py"] if p.is_file()] + sorted((src / "sgi" / name).rglob("*.py"))
    return files


def forbidden_imports(path: Path, src: Path = SRC) -> list[str]:
    package = list(path.relative_to(src).parts[:-1])
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):     # walks into functions too
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):                              # resolves "from ..gt import x"
            base = package[: len(package) - node.level + 1] if node.level else []
            module = ".".join(base + ([node.module] if node.module else []))
            names = [module] + [f"{module}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = [node.value]                                            # importlib.import_module("sgi.gt")
        else:
            continue
        found |= {f"sgi.{f}" for n in names for f in FORBIDDEN if n == f"sgi.{f}" or n.startswith(f"sgi.{f}.")}
    return sorted(found)


def test_inference_modules_do_not_import_harness_modules():
    files = inference_files()
    assert SRC / "sgi" / "types.py" in files and SRC / "sgi" / "solver.py" in files
    assert {str(p.relative_to(SRC)): hits for p in files if (hits := forbidden_imports(p))} == {}


def test_the_checker_sees_every_import_form(tmp_path):
    bad, ok = tmp_path / "sgi" / "stages" / "goal.py", tmp_path / "sgi" / "pipeline.py"
    bad.parent.mkdir(parents=True)
    bad.write_text("import sgi.scene\nfrom sgi import evaluate, config\nfrom ..gt.regions import build_regions\n"
                   "from .. import viz\ndef f():\n    from sgi.synth.render import render\n"
                   "    return __import__('importlib').import_module('sgi.freespace')\n")
    ok.write_text("import numpy\nfrom sgi.config import Config\nfrom .stages import run, goal\nfrom sgi.runner import x\n")
    assert inference_files(tmp_path) == [bad, ok]
    assert forbidden_imports(bad, tmp_path) == [
        "sgi.evaluate", "sgi.freespace", "sgi.gt", "sgi.scene", "sgi.synth", "sgi.viz"]
    assert forbidden_imports(ok, tmp_path) == []
