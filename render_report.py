"""Render the Vietnamese report from measured artifacts, preserving the template."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "evaluation"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def pct(x):
    return f"{100 * float(x):.2f}".replace(".", ",")


def num(x, places=2):
    return f"{x:.{places}f}".replace(".", ",")


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((OUT / "dataset_audit.json").read_text(encoding="utf-8"))
    errors = json.loads((OUT / "error_analysis.json").read_text(encoding="utf-8"))
    benchmark = json.loads((OUT / "benchmark.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(Path("D:/vietfood/runs/detect/vietfood68_5070-2/results.csv").open()))
    classes = list(csv.DictReader((OUT / "test_per_class.csv").open()))
    val, test = summary["splits"]["val"], summary["splits"]["test"]
    op = test["operating_at_val_confidence"]
    mapping = {"TEST_P": pct(test["precision"]), "TEST_R": pct(test["recall"]),
               "TEST_MAP50": pct(test["map50"]), "TEST_MAP": pct(test["map50_95"]),
               "VAL_CONF": num(op["confidence"], 4), "TEST_CONF": num(test["native_confidence_at_best_macro_f1"], 4),
               "OP_P": pct(op["micro_precision"]), "OP_R": pct(op["micro_recall"]),
               "OP_MACRO_P": pct(op["macro_precision"]), "OP_MACRO_R": pct(op["macro_recall"]),
               "MODEL_SHA": summary["model_sha256"], "TEST_VAL_MAP_DIFF": num((test["map50_95"]-val["map50_95"])*100),
               "EVAL_PRE": num(test["speed_ms_per_image"]["preprocess"], 3),
               "EVAL_INFER": num(test["speed_ms_per_image"]["inference"], 3),
               "EVAL_POST": num(test["speed_ms_per_image"]["postprocess"], 3)}
    count = sum(s["images"] for s in audit["splits"].values())
    data_rows = []
    for s, name, token in (("train", "Train", "TRAIN"), ("valid", "Validation", "VAL"), ("test", "Test", "TEST")):
        d = audit["splits"][s]
        data_rows.append([name, f"{d['images']:,}".replace(",", "."), f"{d['labels']:,}".replace(",", "."), f"{d['instances']:,}".replace(",", "."), d["empty_labels"], d["classes_present"]])
        mapping[token + "_SHARE"] = num(d["images"] / count * 100)
        mapping[token + "_INST_PER_IMG"] = num(d["instances"] / d["images"])
    mapping["DATASET_TABLE"] = table(["Tập", "Ảnh", "File nhãn", "Hộp GT", "Nhãn rỗng", "ID có mặt"], data_rows)
    mapping["DATA_GIB"] = num(sum(d["total_image_bytes"] for d in audit["splits"].values()) / 2**30)
    mapping["IMBALANCE"] = num(audit["splits"]["train"]["max_instances_per_class"] / audit["splits"]["train"]["min_instances_per_class"])
    deltas = [float(rows[i]["time"]) - float(rows[i-1]["time"]) for i in range(73, 150)]
    assert len(rows) == 150 and all(d > 0 for d in deltas)
    mapping.update(EPOCH_MIN=num(statistics.median(deltas)/60), EPOCH_MIN_RANGE=num(min(deltas)/60),
                   EPOCH_MAX_RANGE=num(max(deltas)/60), EST_HOURS=num(statistics.median(deltas)*150/3600))
    mapping["EPOCH_TABLE"] = table(["Epoch", "Precision", "Recall", "mAP50", "mAP50–95"],
        [[r["epoch"]] + [pct(r[f"metrics/{k}(B)"]) for k in ("precision", "recall", "mAP50", "mAP50-95")]
         for r in rows if int(r["epoch"]) in (1, 10, 25, 50, 72, 73, 100, 125, 140, 141, 150)])
    mapping["METRICS_TABLE"] = table(["Nguồn", "Ảnh", "Precision", "Recall", "mAP50", "mAP50–95"],
        [["Log epoch 150 (validation)", "6.602"] + [pct(rows[-1][f"metrics/{k}(B)"]) for k in ("precision", "recall", "mAP50", "mAP50-95")]] +
        [[name, f"{s['images']:,}".replace(",", ".")] + [pct(s[k]) for k in ("precision", "recall", "map50", "map50_95")]
         for name, s in (("Validation đánh giá lại", val), ("Test sau training", test))])
    mapping["THRESHOLD_TABLE"] = table(["Confidence", "TP", "FP", "FN", "Precision micro", "Recall micro"],
        [[num(x["confidence"], 4), x["tp"], x["fp"], x["fn"], pct(x["micro_precision"]), pct(x["micro_recall"])] for x in (test["operating_at_025"], op)])
    def class_rows(items):
        return [[r["name"], r["images"], r["instances"]] + [pct(r[k]) for k in ("precision_native", "recall_native", "ap50", "ap50_95")] for r in items]
    headers = ["Lớp", "Ảnh có GT", "Hộp GT", "P", "R", "AP50", "AP50–95"]
    mapping["WEAK_CLASSES"] = table(headers, class_rows(sorted(classes, key=lambda r: float(r["ap50_95"]))[:5]))
    mapping["BEST_CLASSES"] = table(headers, class_rows(sorted(classes, key=lambda r: float(r["ap50_95"]), reverse=True)[:5]))
    mapping["ALL_CLASSES"] = table(headers, class_rows(classes))
    mapping["EXAMPLES_TABLE"] = table(["Mẫu", "File", "GT", "TP", "FP", "FN"],
        [[i+1, f"`{r['file_name']}`", r["gt"], r["tp"], r["fp"], r["fn"]] for i, r in enumerate(errors["examples"])])
    mapping["BENCHMARK_TABLE"] = table(["Chỉ số", "Số đo"],
        [["Số ảnh / warmup", f"{benchmark['images']} / {benchmark['warmup_images']}"],
         ["Forward trung bình / trung vị / p95 (ms)", " / ".join(num(benchmark["speed_ms"]["inference"][k], 3) for k in ("mean", "median", "p95"))],
         ["Wall time trung bình (ms/ảnh)", num(benchmark["wall_ms_per_image"], 3)],
         ["Thông lượng wall time (ảnh/giây)", num(benchmark["wall_images_per_second"], 2)],
         ["CUDA allocated cực đại (MiB)", num(benchmark["peak_cuda_allocated_bytes"]/2**20, 2)]])
    mapping["TEST_STATUS"] = "Đạt 3 kiểm thử" if (ROOT / "reports" / "unit-tests.log").exists() and "OK" in (ROOT / "reports" / "unit-tests.log").read_text(encoding="utf-8") else "Chưa xác nhận lại trong lượt lập báo cáo"
    assert op["tp"] == errors["totals"]["tp"] and op["fp"] == errors["totals"]["fp"] and op["fn"] == errors["totals"]["fn"]
    text = (ROOT / "reports" / "report_template.md").read_text(encoding="utf-8")
    for key, value in mapping.items():
        text = text.replace(f"@@{key}@@", value)
    assert "@@" not in text, "Unfilled report token"
    (ROOT / "Bao_cao_VietFood68.md").write_text(text, encoding="utf-8")
    print(f"Report written: {len(text)} characters, {len(text.splitlines())} lines")


if __name__ == "__main__":
    main()
