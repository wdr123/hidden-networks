import numpy as np 
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib import pyplot 
import matplotlib.pyplot as plt 
from matplotlib import pyplot 
import matplotlib.patches as mpatches

def draw_reliability_graph(ece, bin_acc, bins, prune, arch, loss_type):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    ax.set_axisbelow(True)
    ax.grid(color = 'gray', linestyle = 'dashed')
    plt.bar(bins, bins, width = 0.1, alpha = 0.3, edgecolor = 'black', color = 'r', hatch = '\\')
    plt.bar(bins, bin_acc, width = 0.1, alpha = 0.1, edgecolor = 'black', color = 'b')
    plt.plot([0, 1], [0, 1], '--', color = 'gray', linewidth = 2)
    plt.gca().set_aspect('equal', adjustable = 'box')
    ECE_patch = mpatches.Patch(color = 'green', label = 'ECE: {:.2f}%'.format(ece*100))
    plt.legend(handles = [ECE_patch])
    plt.savefig('hists/ece_plot_'+arch+'_'+spar+'_'+loss_type+'.png', bbox_inches = 'tight')
    plt.clf()

def draw_histogram(conf, gt, pred, prune, arch, n_bins = 10):
    no_correct, no_incorrect = np.zeros(n_bins), np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m/n_bins, (m+1)/n_bins
        for i in range(len(gt)):
            if conf[i]>a and conf[i]<=b:
                if gt[i]==pred[i]:
                    no_correct[m]+=1
                else:
                    no_incorrect[m]+=1

    width = 0.55
    confs = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    ind = np.arange(n_bins)
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(ind, no_correct, width, color = 'r')
    ax.bar(ind, no_incorrect, width, bottom = no_correct, color = 'b')
    ax.set_ylabel('Number of Samples', fontsize = 20)
    ax.set_xlabel('Confidence', fontsize = 20)
    ax.set_xticks(ind, confs, fontsize = 20, rotation = 90)
    ax.legend(labels = ['Correct', 'Incorrect'])
    plt.savefig('hists/correct_incorrect_'+arch+'_'+prune+'.png', bbox_inches = 'tight')
    plt.clf()


def ece_score(gt, pred, conf, n_bins = 10):
    bin_acc, bin_conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m/n_bins, (m+1)/n_bins
        for i in range(len(gt)):
            if conf[i]>a and conf[i]<=b:
                Bm[m]+=1
                if gt[i]==pred[i]:
                    bin_acc[m]+=1
                bin_conf[m]+=conf[i]
        if Bm[m]!=0:
            bin_acc[m] = bin_acc[m]/Bm[m]
            bin_conf[m] = bin_conf[m]/Bm[m]
    
    ece = 0
    acc = 0
    for m in range(n_bins):
        ece+=Bm[m]*np.abs((bin_acc[m]-bin_conf[m]))
        acc+=Bm[m]*bin_acc[m]

    return ece/sum(Bm), acc/sum(Bm), bin_acc, Bm


if __name__=="__main__":
    archs = [ 'cResNet50', 'cResNet101']
    n_bins = 10
    bins = np.linspace(0, 1, 10)
    spar = '0.05'
    n_class = 10
    for arch in archs:
        print("\n Working on an architecture", arch)
        bins = np.linspace(0, 1, 10)
        runs = [1, 2, 3]
        run_outputs = []
        run_acc = []
        run_ece = []
        run_alphas = []
        run_gts = []
        for run in runs:
            in_outputs = np.load("outputs/output_"+arch+'_CIFAR10_'+spar+'_'+str(run)+'.npy')
            gts = np.load('outputs/gts_'+arch+'_CIFAR10_'+spar+'_'+str(run)+'.npy')
            pred = in_outputs.argmax(axis = 1)
            err = sum(np.not_equal(pred, gts).astype(int))/len(gts)
            alpha = np.log((1-err)/err)+np.log(n_class-1)
            run_outputs.append(in_outputs)
            run_alphas.append(alpha)
            run_pred = in_outputs.argmax(axis = 1)
            in_outputs = softmax(in_outputs, axis = 1)
            ece, acc, bin_acc, _ = ece_score(gts, in_outputs.argmax(axis = 1), in_outputs.max(axis = 1), n_bins = len(bins))
            print("Accuracy for a run", run, acc, "ece", ece, "accuracy", acc)
            run_acc.append(acc)
            run_ece.append(ece)
            run_gts.append(gts)
        run_acc = np.array(run_acc)
        run_ece = np.array(run_ece)
        run_gts = np.array(run_gts)
        mean_acc, sd_acc = np.mean(run_acc), np.std(run_acc)
        mean_ece, sd_ece = np.mean(run_ece), np.std(run_ece)
        run_alphas = np.array(run_alphas)
        run_alphas = run_alphas
        run_outputs = np.array(run_outputs)
        total_outputs = (run_outputs.T*run_alphas).T
        total_outputs = np.sum(total_outputs, axis = 0)
        softmax_total_outputs = softmax(total_outputs, axis = 1)
        total_pred = softmax_total_outputs.argmax(axis =1)
        total_conf = softmax_total_outputs.max(axis = 1)
        gt = np.mean(run_gts, axis = 0)
        ece, acc, bin_acc, _ = ece_score(gt, total_pred, total_conf, n_bins = len(bins))
        print("For sparsity:", spar, "ensemble accuracy:", acc, "ece", ece, "run accuracy:", mean_acc, "+-", sd_acc, "ece", mean_ece, "+-", sd_ece )
