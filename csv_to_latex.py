#!/usr/bin/env python3
"""
csv_to_latex.py

Reads averaged CSVs from `averages/<database>/<setting>_averaged.csv` and writes
one LaTeX table per setting. Each table lists every database that contains that
setting; databases missing the setting are omitted from that table.
"""
import os
import glob
import pandas as pd

# config
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
    'imagenet': 'Imagenet',
}

# Default parameters used when not overridden by the setting name.
DEFAULT_C = 0.07
DEFAULT_DELTA = 0.4
DEFAULT_N = 1000


# helpers
def fmt_val(v):
    s = f"{v:.3f}"
    return f"\\textbf{{{s}}}" if v > 0.50 else s


def db_label(name):
    return DB_DISPLAY.get(name, name.replace('_', '-').title())


def parse_setting(filename):
    """Return the setting name from a file named `<setting>_averaged.csv`."""
    base = os.path.basename(filename)
    return base.replace('_averaged.csv', '')


def setting_caption_desc(setting):
    """
    Convert a setting name like 'S95', 'C5_S99', 'C7_S99', 'S95_N2000'
    into a LaTeX-safe caption fragment describing (c, delta), S and N.
    """
    c = DEFAULT_C
    delta = DEFAULT_DELTA
    S = None
    N = DEFAULT_N

    for token in setting.split('_'):
        if token.startswith('C') and token[1:].isdigit():
            c = int(token[1:]) / 100.0
        elif token.startswith('S') and token[1:].isdigit():
            S = int(token[1:]) / 100.0
        elif token.startswith('N') and token[1:].isdigit():
            N = int(token[1:])

    parts = [f"(c, $\\delta$) = ({c:.2f}, {delta:.1f})"]
    if S is not None:
        parts.append(f"S = {S:.2f}")
    parts.append(f"N = {N}")

    return ", ".join(parts)


def discover_files():
    """Return a dict: setting -> [(database, csv_path), ...]."""
    files = sorted(glob.glob("averages/*/*_averaged.csv"))
    settings = {}
    for f in files:
        db = os.path.basename(os.path.dirname(f))
        setting = parse_setting(f)
        settings.setdefault(setting, []).append((db, f))
    return settings


def render_table(setting, db_files):
    """Build the LaTeX table body for one setting."""
    # Load data and discover all B values across databases in this setting
    data = {}
    all_bs = set()
    for db, path in db_files:
        df = pd.read_csv(path)
        data[db] = df
        all_bs.update(df['B'].astype(int).tolist())

    B_vals = sorted(all_bs)
    nB = len(B_vals)
    last_col = 2 + nB          # col 1 = Dataset, col 2 = Method, rest = B's

    # Only keep methods that actually appear in this setting's files
    present_methods = [m for m in METHOD_ORDER
                       if any(m in df.columns for df in data.values())]
    n_methods = len(present_methods)

    lines = []
    lines.append(r"% Requires: \usepackage{booktabs, multirow}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Average extraction recall across seeds (iterative attack, "
        r"certificate-admitted) for " + setting_caption_desc(setting) + r". "
        r"Bold indicates recall $> 0.50$.}"
    )
    lines.append(r"\label{tab:recall_avg_" + setting + r"}")
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

    return "\n".join(lines)


def main():
    settings = discover_files()
    if not settings:
        print("No averaged CSVs found in averages/*/_averaged.csv")
        return

    combined = []
    for setting, db_files in sorted(settings.items()):
        tex = render_table(setting, db_files)
        outname = f"averages_table_{setting}.tex"
        with open(outname, "w") as f:
            f.write(tex)
        print(f"Saved: {outname}")

        combined.append(f"% Setting: {setting}")
        combined.append(tex)
        combined.append("")

    combined_path = "averages_table.tex"
    with open(combined_path, "w") as f:
        f.write("\n".join(combined))
    print(f"Saved: {combined_path}")


if __name__ == '__main__':
    main()
