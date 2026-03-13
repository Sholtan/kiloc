import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def main(run_dir):
    filename = Path(run_dir) / "history.json"
    with open(filename, 'r') as f:
        history = json.load(f)


    train_loss = []
    val_loss = []
    precision = []
    recall = []
    f1 = []

    for _, dct in enumerate(history):
        train_loss.append(dct['train_loss'])
        val_loss.append(dct['val_loss'])
        precision.append(dct['precision'])
        recall.append(dct['recall'])
        f1.append(dct['f1'])


    figures_dir = Path(run_dir) / "figures"
    figures_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots()
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.legend()
    plt.grid()
    fig.savefig(Path(run_dir) / "figures/losses.png", dpi=150, bbox_inches='tight')


    fig, ax = plt.subplots()
    plt.plot(precision, label='precision')
    plt.plot(recall, label='recall')
    plt.plot(f1, label='f1')
    plt.legend()
    plt.grid()
    fig.savefig(Path(run_dir) / "figures/metrics.png", dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_dir')
    args = parser.parse_args()
    main(args.run_dir)