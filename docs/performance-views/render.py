#!/usr/bin/env python3
"""Render the README figures from the archived measurements in history.json; no GPU work."""
from pathlib import Path
import json
import math
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
RED = '#8C0000'
MUTED = '#79696B'
INK = '#292122'
GRID = '#EADFE0'
CARD = '#FAF5F5'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'svg.hashsalt': 'faster-fastwam-crimson', 'font.size': 12})


def save(fig, path):
    target = path.with_suffix('.svg')
    fig.savefig(target, facecolor='white', metadata={'Date': None})
    target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)


def frame(ax, axis='y'):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(length=0, labelsize=11.5, colors=MUTED, pad=9)
    ax.grid(axis=axis, color=GRID, linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def hero():
    data = json.loads((ROOT/'history.json').read_text())['rows'][-1]
    values = [data['reference_ms'], data['candidate_ms']]
    assert all(math.isfinite(v) and v > 0 for v in values)
    fig = plt.figure(figsize=(9, 4.5), facecolor='white')
    fig.text(.06, .88, 'Faster-FastWAM, optimized with an AI agent', color=RED, weight='bold', fontsize=19)
    fig.text(.06, .78, 'FastWAM LIBERO recipe · 8 GPUs · 16 samples per GPU · BF16 mixed precision',
             color=MUTED, fontsize=10.5)
    ax = fig.add_axes([.24, .24, .39, .45])
    ax.barh([1, 0], values, color=['#D7BDBF', RED], height=.24, zorder=3)
    ax.set_yticks([1, 0], ['Baseline', 'Faster-FastWAM'])
    for y, value in zip([1, 0], values):
        ax.text(value+40, y, f'{value:,.1f} ms', va='center', color=INK, fontsize=11.5, weight='bold')
    ax.set_xlim(0, 2100)
    ax.set_ylim(-.6, 1.6)
    ax.set_xticks([0, 500, 1000, 1500])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    frame(ax, 'x')
    ax.tick_params(labelsize=10)
    ax.set_xlabel('Time per training iteration (ms) ↓', color=MUTED, labelpad=12, fontsize=11)
    fig.text(.85, .54, f"{values[0]/values[1]:.2f}×", color=RED, fontsize=35, weight='bold', ha='center')
    fig.text(.85, .43, 'iteration speedup', color=MUTED, fontsize=12, ha='center')
    save(fig, ROOT/'01-benchmark')


def history():
    rows = json.loads((ROOT/'history.json').read_text())['rows']
    note_top = 5.69 + (math.ceil(len(rows)/2)-1)*.74   # inches below the top edge, under the last row of cards
    height = note_top + .62
    top = lambda inches: 1 - inches/height       # figure fraction from inches below the top edge
    fig = plt.figure(figsize=(9, height), facecolor='white')
    fig.text(.09, top(.36), 'Faster-FastWAM', fontsize=11, weight='bold', color=RED)
    fig.text(.09, top(.79), 'Agent Optimization History', fontsize=21.5, weight='bold', color=INK)
    fig.text(.94, top(.65), f"{rows[-1]['speedup']:.2f}×", fontsize=29, ha='right', weight='bold', color=RED)
    fig.text(.94, top(.88), 'iteration speedup', fontsize=10.5, ha='right', color=MUTED)
    fig.text(.09, top(1.31), 'Cumulative speedup · 8 GPUs · LIBERO recipe', fontsize=11, color=MUTED)
    ax = fig.add_axes([.09, top(4.19), .85, 2.61/height])
    x = [row['round'] for row in rows]
    values = [row['speedup'] for row in rows]
    ax.plot(x, values, color=RED, linewidth=2.3, zorder=2)
    for xi, value, row in zip(x, values, rows):
        screen = row['check'] != 'full'
        ax.plot([xi], [value], marker='o', markersize=6.5, zorder=3, color=RED,
                markerfacecolor='white' if screen else RED, markeredgewidth=1.8)
    for i, value in enumerate(values):
        last = i == len(rows)-1
        ax.annotate(f'{value:.2f}×', (x[i], value), xytext=(0, 11 if i % 2 == 0 or last else -23),
                    textcoords='offset points', ha='center', fontsize=12,
                    color=RED if last else INK, weight='bold')
    ax.set_xticks(x, [f"R{step:02d}" for step in x])
    ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.1f}×'))
    ax.set_ylim(1.0, 3.8)
    ax.set_xlim(x[0]-.6, x[-1]+.6)
    frame(ax)
    for i, row in enumerate(rows):
        bx, by = .09+(i % 2)*.44, top(5.18+(i // 2)*.74)
        fig.add_artist(FancyBboxPatch((bx, by-.378/height), .41, .585/height,
                       boxstyle='round,pad=0.004,rounding_size=0.007',
                       transform=fig.transFigure, facecolor=CARD, edgecolor='none'))
        fig.text(bx+.014, by-.08/height, f"R{row['round']:02d}", fontsize=12, weight='bold', color=RED)
        label = textwrap.fill(row['label'], width=24, break_long_words=False, break_on_hyphens=False)
        fig.text(bx+.079, by-.08/height, label, fontsize=11.5, color=INK,
                 weight='bold', va='center', linespacing=1.35)
    fig.text(.09, top(note_top), 'Open marker: timing screen. Filled: full check (all four gates).\nCumulative '
             'configurations, not isolated ablations.', fontsize=9.5, color=MUTED, linespacing=1.4, va='top')
    save(fig, ROOT/'02-history')


def modulemap():
    fig = plt.figure(figsize=(9, 6.6), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(.055, .92, 'What changed in Faster-FastWAM', fontsize=23, color=INK, weight='bold')
    cards = [
        ('Data pipeline', 'Decode only the frames the model uses\nColumn tensors · uint8 frames · T5 memo',
         'datasets/lerobot/base_lerobot_dataset.py\nlerobot/lerobot_dataset.py · video_utils.py\n'
         'robot_video_dataset.py', '4 files'),
        ('VAE encode', 'Inductor instead of the CUDA-graph backend\nFused norm / activation chains',
         'models/wan22/fastwam.py', '1 file'),
        ('MoT layers', "Upstream's regional compile of the joint\nvideo / action layers, switched on",
         'models/wan22/fastwam.py\n(compiled layer: models/wan22/mot.py, upstream)', '1 file'),
        ('Host synchronization', 'Loss dict on the GPU · metrics gathered only\non logging steps · non-blocking copies',
         'trainer.py\nmodels/wan22/fastwam.py', '2 files'),
    ]
    for i, (title, description, paths, count) in enumerate(cards):
        x, y = .055+(i % 2)*.455, .49-(i // 2)*.39
        ax.add_patch(FancyBboxPatch((x, y), .425, .34,
                     boxstyle='round,pad=0.009,rounding_size=0.012', facecolor=CARD, edgecolor=GRID))
        ax.plot([x, x+.425], [y+.34, y+.34], color=RED, linewidth=2, solid_capstyle='round')
        ax.text(x+.014, y+.30, f'{i+1:02d}', fontsize=11, color=RED, weight='bold')
        ax.text(x+.41, y+.30, count, fontsize=10, color=MUTED, ha='right')
        ax.text(x+.014, y+.254, title, fontsize=13, color=INK, weight='bold')
        ax.text(x+.014, y+.198, description, fontsize=10.3, color=INK, va='top', linespacing=1.6)
        ax.text(x+.014, y+.035, paths, fontsize=9.5, color=MUTED, linespacing=1.5)
    save(fig, ROOT.parent/'change-views/module-map')


if __name__ == '__main__':
    hero()
    history()
    modulemap()
