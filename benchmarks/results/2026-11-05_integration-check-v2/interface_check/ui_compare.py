"""Compares what the browser showed and what the downloaded Excel holds with what the API returned and the model generated."""
import glob
import json
import sys
from pathlib import Path

from openpyxl import load_workbook

out = Path(sys.argv[1])
ui = json.loads((out / "ui_report.json").read_text("utf-8"))
api = [json.loads(line) for line in (out / "api_analysis.jsonl").read_text("utf-8").splitlines()][-1]
gens = [json.loads(line) for line in (out / "generations.jsonl").read_text("utf-8").splitlines()]
books = sorted(glob.glob(str(out / "*.xlsx")))
rows = list(load_workbook(books[-1])["Resultados"].values)
excel = [dict(zip(rows[0], r)) for r in rows[1:]]


def norm(text):
    return " ".join((text or "").split())


frags = []
for res, text, cells in zip(api["resultados"], ui["fragments_text"], excel):
    a = res["analisis"]
    t = norm(text)
    citas = [e["cita"] for e in a["evidencia"] if e["cita"]]
    frags.append({"fragment": res["fragmento"], "state": a["estado"], "label": a["etiqueta"], "label_text": a["etiqueta_texto"],
                  "ui_label": norm(a["etiqueta_texto"]) in t,
                  "ui_explanation": norm(a["explicacion"] or a["motivo"]) in t,
                  "ui_quotes": all(norm(c) in t for c in citas),
                  "ui_passages": all(norm(p)[:60] in t for p in res["relevant_chunks"]),
                  "ui_no_internal_label": not any(x in t for x in ("insufficient_evidence", "contradicted", "supported")),
                  "excel_label": cells["Análisis del modelo"] == a["etiqueta_texto"],
                  "excel_explanation": cells["Explicación del modelo"] == a["explicacion"],
                  "excel_quotes": all(c in (cells["Evidencia citada"] or "") for c in citas),
                  "excel_reason": cells["Motivo sin resultado"] == a["motivo"]})

ok_fragments = all(all(v for k, v in f.items() if k.startswith(("ui_", "excel_"))) for f in frags)
checks = {"default_off": ui["analysis_checkbox_default"] is False, "switched_on_by_click": ui["analysis_checkbox_after_click"] is True,
          "generations_equal_fragments": len(gens) == len(api["resultados"]), "browser_errors": ui["errors"],
          "think_false": all(g["think"] is False for g in gens), "num_thread": sorted({g["options"].get("num_thread") for g in gens}),
          "model": sorted({g["model"] for g in gens}), "fragments_compared": len(frags), "ui_wait_s": ui["ui_wait_s"], "fragments": frags,
          "all_ok": len(gens) == len(api["resultados"]) == len(frags) == len(excel) and not ui["errors"] and ok_fragments}
(out / "ui_check.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), "utf-8")
print(json.dumps({k: v for k, v in checks.items() if k != "fragments"}, ensure_ascii=False))
for f in frags:
    print(f)
sys.exit(0 if checks["all_ok"] else 1)
