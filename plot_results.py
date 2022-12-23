import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.ticker import MaxNLocator

import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--complete', type=str, help='path to the complete/full scan results')
parser.add_argument('-r', '--reduced', type=str, help='path to the reduced scan results')
args = parser.parse_args()

full_scan_file = args.complete
reduced_scan_file = args.reduced

full_scan_results = pd.read_csv(full_scan_file)
reduced_scan_results = pd.read_csv(reduced_scan_file)

num_epochs = max(full_scan_results.epoch.max(), reduced_scan_results.epoch.max())

def plot_auc():
    plt.plot(full_scan_results.epoch, full_scan_results['test_auc'], label='full scan')
    plt.plot(reduced_scan_results.epoch, reduced_scan_results['test_auc'], label='reduced scan')

    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    # if num_epochs < MAX_EPOCHS:
    #     plt.xticks(range(num_epochs))
    EXT = '.png'
    stem = re.match(r'.*results(_.*).csv', full_scan_file).group(1) + '_full+reduced' + EXT
    plt.savefig(f'results_auc{stem}')
    plt.close()

def plot_comparisons(metric):
    if metric == 'acc':
        metric_out = 'Accuracy'
    elif metric == 'loss':
        metric_out = 'Loss'
    elif metric == 'auc':
        plot_auc()
        return
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    ax[0].plot(full_scan_results.epoch, full_scan_results[f'train_{metric}'], label='full scan train')
    ax[1].plot(full_scan_results.epoch, full_scan_results[f'test_{metric}'], label='full scan test')
    ax[0].plot(reduced_scan_results.epoch, reduced_scan_results[f'train_{metric}'], label='reduced scan train')
    ax[1].plot(reduced_scan_results.epoch, reduced_scan_results[f'test_{metric}'], label='reduced scan test')
    ax[0].set_xlabel('Epoch')
    ax[1].set_xlabel('Epoch')
    ax[0].set_ylabel(metric_out)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title('Train')
    ax[1].set_title('Test')

    max_y = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1])
    ax[0].set_ylim(0,top=max_y)
    ax[1].set_ylim(0,top=max_y)

    fig.tight_layout()

    EXT = '.png'
    stem = re.match(r'.*results(_.*).csv', full_scan_file).group(1) + '_full+reduced' + EXT
    plt.savefig(f'results_{metric}{stem}')
    plt.close()


for metric in ['acc', 'loss', 'auc']:
    plot_comparisons(metric)


def plot():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='results file')
    #results_file = 'train_results_50-samples_5-epochs.csv'
    #results_file = 'train_kspace_results_100-prcnt-scan_50-samples_5-epochs.csv'
    args = parser.parse_args()
    results_file = args.file
    #results_file = 'train_results_50-samples_100-epochs_no-test-transform.csv'

    EXT = '.png'
    stem = re.match(r'.*results(_.*).csv', results_file).group(1) + EXT
    #print('stem is', stem)
    #stem = '_50-samples_100-epochs_no-test-transform.png'
    #stem = '_kspace_100prcnt-scan_50-samples_5-epochs.png'
    #stem = '_50-samples_5-epochs.png'
    results = pd.read_csv(results_file)

    MAX_EPOCHS = 20

    num_epochs = results.epoch.max()

    plt.plot(results.epoch, results.train_loss, label='train')
    plt.plot(results.epoch, results.test_loss, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if num_epochs < MAX_EPOCHS:
        plt.xticks(range(num_epochs))
    plt.savefig('results_loss'+stem)
    plt.close()

    plt.plot(results.epoch, results.train_acc, label='train')
    plt.plot(results.epoch, results.test_acc, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    if num_epochs < MAX_EPOCHS:
        plt.xticks(range(num_epochs))
    plt.legend()
    plt.savefig('results_acc'+stem)
    plt.close()

    plt.plot(results.epoch, results.test_auc)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    if num_epochs < MAX_EPOCHS:
        plt.xticks(range(num_epochs))
    plt.savefig('results_auc'+stem)
    plt.close()
