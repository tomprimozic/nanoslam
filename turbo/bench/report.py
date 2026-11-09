'''
benchmark report: one self-contained HTML page from the run directories (turbo/run.py and bench.py outputs), every
system x sequence, the median over start frames, switchable metric, colour coded per sequence

  python turbo/bench/report.py RUN_DIR [RUN_DIR ...] --out report.html
  (each RUN_DIR holds <system>/<sequence>/metrics.json, one directory per start frame)
'''
import argparse
import json
from pathlib import Path

ORDER = ['turbo', 'orbslam2', 'orbslam2_vo', 'orbslam3', 'dso', 'dso_photo', 'ov2slam', 'ov2slam_vo']
FAMILIES = [('MH_', 'EuRoC'), ('V1_', 'EuRoC'), ('V2_', 'EuRoC'), ('freiburg1', 'TUM fr1'), ('freiburg2', 'TUM fr2/3'),
            ('freiburg3', 'TUM fr2/3'), ('icl', 'ICL-NUIM / ETH3D'), ('eth3d', 'ICL-NUIM / ETH3D'),
            ('tartanair', 'TartanAir'), ('abandoned', 'TartanAir'), ('hospital', 'TartanAir'), ('KITTI', 'KITTI'),
            ('monoVO', 'TUM monoVO')]
METRICS = {   # key: (label, section, field)
  'live_final': ('live final drift (initial-section aligned, rolling)', 'online', 'final'),
  'live_ate': ('live ATE (fully aligned)', 'online', 'ate'),
  'hind_final': ('hindsight final drift (initial-section aligned)', 'final', 'final'),
  'hind_ate': ('hindsight ATE (fully aligned)', 'final', 'ate'),
}


def family(name):
  return next((f for p, f in FAMILIES if name.startswith(p)), 'other')


def collect(run_dirs):
  data = {}
  systems = set()
  for r in run_dirs:
    for f in sorted(Path(r).glob('*/*/metrics.json')):
      system, name = f.parent.parent.name, f.parent.name
      if system not in ORDER:
        continue
      m = json.loads(f.read_text())
      systems.add(system)
      cell = data.setdefault(name, {}).setdefault(system, {k: [] for k in METRICS} | {'tracked': [], 'start': []})
      tracked = m['online'].get('tracked', 0)
      for k, (_, sec, field) in METRICS.items():
        v = m[sec].get(field)
        cell[k].append(None if (not tracked or v is None or v != v or v == float('inf')) else v)
      cell['tracked'].append(tracked)
      cell['start'].append(m.get('start', 0))
  fam_order = list(dict.fromkeys(f for _, f in FAMILIES))
  rows = [{'name': n, 'family': family(n), 'cells': data[n]} for n in
          sorted(data, key=lambda n: (fam_order.index(family(n)) if family(n) in fam_order else 99, n))]
  return [s for s in ORDER if s in systems], rows


PAGE = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>nanoslam turbo benchmark</title>
<style>
body { font: 13px -apple-system, Helvetica, Arial, sans-serif; margin: 20px; color: #222; }
h1 { font-size: 18px; margin: 0 0 6px; }
.controls { margin: 10px 0 14px; display: flex; gap: 22px; flex-wrap: wrap; align-items: center; }
.controls label { margin-right: 10px; cursor: pointer; }
table { border-collapse: collapse; }
th, td { padding: 3px 8px; text-align: right; border-bottom: 1px solid #eee; white-space: nowrap; }
th { position: sticky; top: 0; background: #fff; border-bottom: 2px solid #999; z-index: 1; }
td.name, th.name { text-align: left; }
tr.fam td { background: #f3f3f3; font-weight: 600; border-top: 2px solid #999; }
tr.fam td.name { cursor: pointer; }
tr.summary td { font-weight: 600; background: #fafafa; }
td.fail { color: #b00; font-weight: 600; }
.small { color: #777; font-size: 11px; }
.legend span { display: inline-block; padding: 2px 8px; margin-right: 2px; }
.hidden { display: none; }
</style></head><body>
<h1>nanoslam turbo vs baselines (monocular)</h1>
<div class="small" id="meta"></div>
<div class="controls">
  <div><b>metric</b><br><span id="metrics"></span></div>
  <div><b>colour</b><br>
    <label><input type="radio" name="colour" value="best" checked> ratio to the best system of the sequence</label>
    <label><input type="radio" name="colour" value="turbo"> ratio to turbo</label></div>
  <div><b>value</b><br>
    <label><input type="radio" name="agg" value="median" checked> median over starts</label>
    <label><input type="radio" name="agg" value="min"> best start</label>
    <label><input type="radio" name="agg" value="max"> worst start</label></div>
</div>
<div class="legend small" id="legend"></div>
<table id="t"></table>
<p class="small">Errors in metres (ground-truth units; TUM monoVO: arbitrary scale). Median over the start frames (hover a
cell for the individual runs and the tracked fraction). Failures (nothing tracked / no initialization) count as worst.
Family rows: geometric mean of system / turbo over the sequences both succeed on (&gt; 1: turbo better),
[sequences turbo is better / worse], failures. Click a family row to fold it.</p>
<script>
const DATA = __DATA__;
const METRICS = __METRICS__;
let metric = 'live_final', colour = 'best', agg = 'median';
const folded = new Set();
const ms = document.getElementById('metrics');
for (const [k, label] of Object.entries(METRICS))
  ms.insertAdjacentHTML('beforeend', `<label><input type="radio" name="metric" value="${k}" ${k == metric ? 'checked' : ''}> ${label}</label><br>`);
document.querySelectorAll('input[name=metric]').forEach(e => e.onchange = () => { metric = e.value; draw(); });
document.querySelectorAll('input[name=colour]').forEach(e => e.onchange = () => { colour = e.value; draw(); });
document.querySelectorAll('input[name=agg]').forEach(e => e.onchange = () => { agg = e.value; draw(); });
document.getElementById('meta').textContent = `${DATA.rows.length} sequences, generated ${DATA.generated}`;

function aggregate(vals) {
  if (!vals || !vals.length) return undefined;           // not run
  const v = vals.map(x => x === null ? Infinity : x).sort((a, b) => a - b);
  if (agg == 'min') return v[0];
  if (agg == 'max') return v[v.length - 1];
  const n = v.length;
  return n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}
// ratio -> colour: 1 green, 1.5 yellow-green, 3 orange, >= 10 red
function shade(r) {
  if (!isFinite(r)) return '#e8a0a0';
  const x = Math.min(Math.log10(Math.max(r, 1)) / Math.log10(10), 1);       // 0 .. 1
  const hue = 120 * (1 - x);
  return `hsl(${hue}, 65%, ${88 - 10 * x}%)`;
}
function shadeTurbo(r) {   // < 1 baseline better than turbo (red), > 1 turbo better (green)
  if (!isFinite(r)) return r > 0 ? 'hsl(120,65%,78%)' : '#e8a0a0';
  const x = Math.max(-1, Math.min(1, Math.log10(r)));
  return x >= 0 ? `hsl(120, 60%, ${92 - 14 * x}%)` : `hsl(0, 70%, ${92 + 14 * x}%)`;
}
function fmt(v) {
  if (v === undefined) return '';
  if (!isFinite(v)) return 'fail';
  return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toFixed(3);
}
function legend() {
  const el = document.getElementById('legend');
  if (colour == 'best')
    el.innerHTML = 'ratio to the best: ' + [1, 1.25, 1.6, 2.5, 4, 10].map(r => `<span style="background:${shade(r)}">${r}x</span>`).join('') + `<span style="background:${shade(Infinity)}">fail</span>`;
  else
    el.innerHTML = 'system / turbo: ' + [0.1, 0.3, 0.6, 1, 1.6, 3, 10].map(r => `<span style="background:${shadeTurbo(r)}">${r}</span>`).join('') + ' (red: better than turbo, green: worse)';
}
function draw() {
  legend();
  const S = DATA.systems;
  let h = '<tr><th class="name">sequence</th>' + S.map(s => `<th>${s}</th>`).join('') + '</tr>';
  const fams = [...new Set(DATA.rows.map(r => r.family))];
  const all = [];
  for (const fam of fams) {
    const rows = DATA.rows.filter(r => r.family == fam);
    h += famRow(fam, rows, S);
    for (const r of rows) {
      const v = S.map(s => aggregate(r.cells[s] && r.cells[s][metric]));
      const best = Math.min(...v.filter(x => x !== undefined));
      const t = v[0];
      h += `<tr class="${folded.has(fam) ? 'hidden' : ''}"><td class="name">${r.name}</td>`;
      S.forEach((s, j) => {
        const c = r.cells[s];
        if (!c) { h += '<td></td>'; return; }
        const runs = c[metric].map((x, i) => `start ${c.start[i]}: ${x === null ? 'fail' : x.toFixed(4)} (tracked ${(100 * c.tracked[i]).toFixed(0)}%)`).join('\n');
        const bg = colour == 'best' ? shade(isFinite(v[j]) ? v[j] / best : Infinity)
                                    : (j == 0 ? '#fff' : shadeTurbo(v[j] / t));
        h += `<td class="${isFinite(v[j]) ? '' : 'fail'}" style="background:${bg}" title="${runs}">${fmt(v[j])}<span class="small"> ${c[metric].length < 3 ? '(' + c[metric].length + ')' : ''}</span></td>`;
      });
      h += '</tr>';
      all.push(r);
    }
  }
  h += famRow('all', all, S, true);
  document.getElementById('t').innerHTML = h;
  document.querySelectorAll('tr.fam td.name').forEach(e => e.onclick = () => {
    const f = e.dataset.fam; folded.has(f) ? folded.delete(f) : folded.add(f); draw(); });
}
function famRow(fam, rows, S, summary) {
  let h = `<tr class="${summary ? 'fam summary' : 'fam'}"><td class="name" data-fam="${fam}">${folded.has(fam) ? '+ ' : ''}${fam} (${rows.length})</td>`;
  S.forEach((s, j) => {
    if (j == 0) { const f = rows.filter(r => !isFinite(aggregate(r.cells.turbo && r.cells.turbo[metric]))).length; h += `<td>${f ? 'fail ' + f : ''}</td>`; return; }
    let logs = [], better = 0, worse = 0, fail = 0, n = 0;
    for (const r of rows) {
      if (!r.cells[s]) continue;
      n++;
      const a = aggregate(r.cells.turbo && r.cells.turbo[metric]), b = aggregate(r.cells[s][metric]);
      if (!isFinite(b)) { fail++; continue; }
      if (!(isFinite(a) && a > 0 && b > 0)) continue;
      logs.push(Math.log(b / a)); if (a < b) better++; else if (a > b) worse++;
    }
    if (!n) { h += '<td></td>'; return; }
    const g = Math.exp(logs.reduce((x, y) => x + y, 0) / logs.length);
    h += `<td style="background:${shadeTurbo(g)}" title="geometric mean ${s} / turbo over ${logs.length} sequences">${g.toFixed(2)} <span class="small">[${better}/${worse}]${fail ? ' f' + fail : ''}</span></td>`;
  });
  return h + '</tr>';
}
draw();
</script></body></html>
'''


def main():
  p = argparse.ArgumentParser()
  p.add_argument('runs', nargs='+', type=Path)
  p.add_argument('--out', type=Path, default=Path('report.html'))
  a = p.parse_args()
  systems, rows = collect(a.runs)
  import datetime
  data = {'systems': systems, 'rows': rows, 'generated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
  html = PAGE.replace('__DATA__', json.dumps(data)).replace('__METRICS__', json.dumps({k: v[0] for k, v in METRICS.items()}))
  a.out.write_text(html)
  print(f'{a.out}: {len(rows)} sequences, systems {systems}')


if __name__ == '__main__':
  main()
