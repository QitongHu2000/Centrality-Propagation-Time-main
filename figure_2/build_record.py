"""Record local figure inputs and publish copies to the maintained documents."""

import hashlib
import json
import re
import shutil

from paths import ASSETS, CODE, FIGURE, REPORTS, figure_path


def prepare():
    (REPORTS / "verification").mkdir(parents=True, exist_ok=True)


def sync_documents():
    """Copy referenced PDFs when this figure lives in the project code directory."""
    if CODE.parent.name != "code":
        return []
    project = CODE.parent.parent
    sources = [figure_path()]
    if FIGURE == "figure_1":
        sources.append(ASSETS / "graphical_abstract.pdf")
    synced = []
    for name in ("main_final", "response_letter"):
        folder = project / name
        if not (folder / f"{name}.tex").is_file():
            continue
        referenced = set()
        for tex in folder.glob("*.tex"):
            referenced.update(
                re.findall(
                    r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}",
                    tex.read_text(),
                )
            )
        for source in sources:
            if source.name not in referenced:
                continue
            target = folder / "figures" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.read_bytes() != source.read_bytes():
                shutil.copy2(source, target)
            synced.append(str(target.relative_to(project)))
    return synced


def record(inputs):
    def hashes(paths):
        return {
            str(path.resolve().relative_to(CODE)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(paths)
        }

    report = {
        "figure": FIGURE,
        "entrypoint": "plot.py",
        "inputs": hashes(inputs),
        "sources": hashes(CODE.glob("*.py")),
        "outputs": hashes([figure_path(), figure_path(suffix=".png")]),
        "scope": "All inputs and programs are local; plotting does not run simulations.",
    }
    (CODE / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    synced = sync_documents()
    print(f"Built {figure_path()}; synchronized {len(synced)} document figure copies")


if __name__ == "__main__":
    synced = sync_documents()
    print(f"Synchronized {len(synced)} document figure copies")
