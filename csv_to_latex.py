#!/usr/bin/env python3
"""
csv_to_latex.py

Reads averaged CSVs from `averages/` and emits a single compact LaTeX table.
"""
import os
import glob
import pandas as pd

# ------------------------------------------------------------------ config
METHOD_ORDER = [
    'soliton_free_independent',
    'soliton_free_mirrored',
    'soliton_data_independent',
    'soliton_data_mirrored',
    'trap_weights_independent',
    'trap_weights_mirrored',
    'passive', 
]

METHOD_NAMES = {
    'soliton_free_independent': 'SF-Ind',
    'soliton_free_mirrored': 'SF-Mir',
    'soliton_data_independent': 'SD-Ind',
    'soliton_data_mirrored': 'SD-Mir',
    'trap_weights_independent': 'TW-Ind',
    'trap_weights_mirrored': 'TW-Mir',
    'passive': 'Passive',
}

# Visual groups: thin rule inserted after each group except the last
GROUPS = [
    ('soliton_free_independent', 'soliton_free_mirrored'),
    ('soliton_data_independent', 'soliton_data_mirrored'),
    ('trap_weights_independent', 'trap_weights_mirrored'),
    ('passive',),
]

DB_DISPLAY = {
    'mnist': 'MNIST',
    'fashion_mnist': 'Fashion-MNIST',
    'emnist': 'EMNIST',
    'svhn': 'SVHN',
    'cifar10': 'CIFAR-10',
    'cifar100': 'CIFAR-100',
    'harus': 'HARUS',
    'imagenet':'Imagenet',
}

# ------------------------------------------------------------------ helpers
def fmt_val(v):
    s = f"{v:.3f}"
    return f"\\textbf{{{s}}}" if v > 0.50 else s


def db_label(name):
    return DB_DISPLAY.get(name, name.replace('_', '-').title())



def main():
    files = sorted(glob.glob("averages/*_averaged.csv"))
    if not files:
        print("No averaged CSVs found in averages/")
        return

    # Load data and discover all B values
    data = {}
    all_bs = set()
    for f in files:
        db = os.path.basename(f).replace('_averaged.csv', '')
        df = pd.read_csv(f)
        data[db] = df
        all_bs.update(df['B'].astype(int).tolist())

    B_vals = sorted(all_bs)
    nB = len(B_vals)
    last_col = 2 + nB          # col 1 = Dataset, col 2 = Method, rest = B's

    # Only keep methods that actually appear in the files
    present_methods = [m for m in METHOD_ORDER
                       if any(m in df.columns for df in data.values())]
    n_methods = len(present_methods)

    lines = []
    lines.append(r"% Requires: \usepackage{booktabs, multirow}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Average extraction recall across seeds (iterative attack, certificate-admitted). Bold indicates recall $> 0.50$.}")
    lines.append(r"\label{tab:recall_avg}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.85}")
    lines.append(r"\begin{tabular}{@{}ll*{" + str(nB) + r"}{c}@{}}")
    lines.append(r"\toprule")

    # Header
    hdr = r"\textbf{Dataset} & \textbf{Method}"
    for b in B_vals:
        hdr += f" & \\textbf{{{b}}}"
    hdr += r" \\"
    lines.append(hdr)
    lines.append(r"\midrule")

    # Body rows
    first_db = True
    for db in sorted(data.keys()):
        df = data[db]
        rows_by_b = {int(row['B']): row for _, row in df.iterrows()}

        if not first_db:
            lines.append(r"\midrule")
        first_db = False

        for i, method in enumerate(present_methods):
            if method not in df.columns:
                continue

            # Dataset name only on the first row of the block
            cell1 = (r"\multirow{" + str(n_methods) + r"}{*}{\textbf{"
                     + db_label(db) + r"}}") if i == 0 else ""
            method_name = METHOD_NAMES.get(method, method)

            row = f"{cell1} & {method_name}"
            for b in B_vals:
                val = rows_by_b[b][method] if b in rows_by_b else float('nan')
                row += f" & {fmt_val(val)}" if b in rows_by_b else " & ---"
            row += r" \\"
            lines.append(row)

            # Thin separator after each group (except the very last row)
            for grp in GROUPS:
                if method == grp[-1] and method != present_methods[-1]:
                    lines.append(r"\cmidrule[0.3pt](lr){2-" + str(last_col) + r"}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    with open("averages_table.tex", "w") as f:
        f.write(tex)
    print("Saved: averages_table.tex")

if __name__ == '__main__':
    main()