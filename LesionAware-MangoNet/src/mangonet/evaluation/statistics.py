"""Aggregate seed-matched results, confidence intervals and Holm correction."""
from collections import defaultdict
from pathlib import Path
import pandas as pd
from mangonet.evaluation.metrics import stats_from_seeds, paired_significance, holm_adjust
from mangonet.utils.reproducibility import save_json


def aggregate_results(results, output_dir, metric='macro_f1', comparator=None):
    """results: list of dicts with source, target, seed, metrics.

    The result list should contain exactly one record per configuration per seed;
    no values are filled in for missing experiments.
    """
    groups = defaultdict(list)
    for row in results:
        name = row.get('model', 'LesionAwareMangoNet')
        key = (name, row['source'], row['target'])
        groups[key].append(row)
    summary, comparisons, rows = [], [], []
    for key, items in sorted(groups.items()):
        stats = stats_from_seeds([record['metrics'][metric] for record in items])
        s = {'model': key[0], 'source': key[1], 'target': key[2], 'metric': metric, **stats}
        summary.append(s)
        for item in items:
            rows.append({'model': key[0], 'source': key[1], 'target': key[2],
                         'seed': item['seed'], metric: item['metrics'][metric]})
    if comparator:
        for key, items in groups.items():
            if key[0] == comparator:
                continue
            comp_key = (comparator, key[1], key[2])
            if comp_key not in groups:
                continue
            own = {v['seed']: v['metrics'][metric] for v in items}
            comp = {v['seed']: v['metrics'][metric] for v in groups[comp_key]}
            common = sorted(set(own) & set(comp))
            if len(common) < 2:
                continue
            p = paired_significance([comp[i] for i in common], [own[i] for i in common])
            comparisons.append({'model': key[0], 'comparator': comparator,
                                'source': key[1], 'target': key[2], **p})
        if comparisons:
            adjusted = holm_adjust([r['p_value'] for r in comparisons])
            for item, p in zip(comparisons, adjusted):
                item['holm_p_value'] = p
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / 'seed_summary.csv', index=False)
    pd.DataFrame(rows).to_csv(out / 'seed_level_metrics.csv', index=False)
    if comparisons:
        pd.DataFrame(comparisons).to_csv(out / 'paired_tests.csv', index=False)
    save_json({'summary': summary, 'paired_tests': comparisons,
               'warning': 'Five independent runs are required for the manuscript statistical protocol.'},
              out / 'statistical_summary.json')
    return summary
