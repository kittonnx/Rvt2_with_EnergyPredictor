# save as: compute_energy_minmax.py
import argparse
import json
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path", help="energy_dataset.npz へのパス")
    ap.add_argument("-o", "--out", default="energy_minmax.json",
                    help="出力JSONパス（既定: energy_minmax.json）")
    ap.add_argument("--task", default=None,
                    help="特定タスクのみ計算したい場合に指定（例: sweep_to_dustpan_of_size）")
    args = ap.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)

    # 必須フィールドの存在確認
    if "task_name" not in data or "energy" not in data:
        raise KeyError("NPZに 'task_name' と 'energy' が必要です。")

    task_names = data["task_name"]
    energies   = data["energy"].astype(float).reshape(-1)

    # 文字列化の保険（dtypeがobject/<Uの場合でもOKに）
    task_names = task_names.astype(str).reshape(-1)
    assert len(task_names) == len(energies), "task_name と energy の長さが一致しません。"

    # 集計
    results = {}
    unique_tasks = [args.task] if args.task else sorted(np.unique(task_names))

    for t in unique_tasks:
        mask = (task_names == t)
        if not np.any(mask):
            continue
        e = energies[mask]
        # NaN を除去
        e = e[~np.isnan(e)]
        if e.size == 0:
            continue

        emin = float(np.min(e))
        emax = float(np.max(e))

        results[t] = {
            "emin": emin,
            "emax": emax,
            # 参考情報（使わなくても可）
            "count": int(e.size),
            "mean": float(np.mean(e)),
            "std": float(np.std(e))
        }

    if len(results) == 0:
        raise ValueError("指定条件に一致するタスクが見つかりませんでした。")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✔ wrote: {args.out}")
    for t, v in results.items():
        print(f"{t}: emin={v['emin']:.6f}, emax={v['emax']:.6f} (n={v['count']})")

if __name__ == "__main__":
    main()
