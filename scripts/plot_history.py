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

    # pos metrics
    precision_pos = []
    recall_pos = []
    f1_pos = []

    # neg metrics
    precision_neg = []
    recall_neg = []
    f1_neg = []

    for _, dct in enumerate(history):
        train_loss.append(dct['train_loss'])
        val_loss.append(dct['val_loss'])
        precision.append(dct['precision'])
        recall.append(dct['recall'])
        f1.append(dct['f1'])

        precision_pos.append(dct['precision_pos'])
        recall_pos.append(dct['recall_pos'])
        f1_pos.append(dct['f1_pos'])

        precision_neg.append(dct['precision_neg'])
        recall_neg.append(dct['recall_neg'])
        f1_neg.append(dct['f1_neg'])


    figures_dir = Path(run_dir) / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Plot losses
    fig, ax = plt.subplots()
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.legend()
    plt.grid()
    fig.savefig(Path(run_dir) / "figures/losses.png", dpi=150, bbox_inches='tight')

    # Plot overall metrics
    fig, ax = plt.subplots()
    plt.plot(precision, label='precision')
    plt.plot(recall, label='recall')
    plt.plot(f1, label='f1')
    plt.legend()
    plt.grid()
    fig.savefig(Path(run_dir) / "figures/metrics.png", dpi=150, bbox_inches='tight')


    # Plot metrics for positive
    fig, ax = plt.subplots()
    plt.plot(precision_pos, label='precision_pos')
    plt.plot(recall_pos, label='recall_pos')
    plt.plot(f1_pos, label='f1_pos')
    plt.legend()
    plt.grid()
    fig.savefig(Path(run_dir) / "figures/metrics_pos.png", dpi=150, bbox_inches='tight')

    # Plot metrics for negative
    fig, ax = plt.subplots()
    plt.plot(precision_neg, label='precision_neg')
    plt.plot(recall_neg, label='recall_neg')
    plt.plot(f1_neg, label='f1_neg')
    plt.legend()
    plt.grid()
    fig.savefig(Path(run_dir) / "figures/metrics_neg.png", dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_dir')
    args = parser.parse_args()
    main(args.run_dir)