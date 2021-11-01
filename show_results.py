
from utils import *
import os
from collections import defaultdict

label_font = {'size':16, }

def compute_average_accuracy(metric_result, epoch):
    average_accuracies = []
    forgetting_metrics = []
    final_epoch_accuracies = np.array(metric_result[epoch-1::epoch])
    for task_p, task_accuracy in enumerate(final_epoch_accuracies):
        average_accuracy = np.mean(task_accuracy[:task_p+1])
        average_accuracies.append(average_accuracy)
        if task_p > 0:
            forgetting_list = []
            for i in range(task_p):
                forgetting_list += [np.max(final_epoch_accuracies[:task_p,i]) - task_accuracy[i]]
            forgetting_metrics.append(np.mean(forgetting_list))
    return average_accuracies, forgetting_metrics

def plot_acc_lines(data_dict, linestyle_dict, legend_dict):
    plt.ylabel('average accuracy', label_font)
    plt.xlabel('numbers of classes', label_font)
    for model, data in data_dict.items():
        data = np.array(data)
        xs = np.arange(1, data.shape[1] + 1) * 10
        ys = data.mean(axis=0)
        plt.xticks(xs)
        plt.plot(xs, ys, linestyle_dict[model], label=legend_dict[model], marker='o', markersize=3)
        fs = data.std(axis=0)
        plt.fill_between(xs, ys-fs, ys+fs, color=linestyle_dict[model][0], alpha=0.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.show()

def plot_fgt_lines(data_dict, linestyle_dict, legend_dict):
    plt.ylabel('average forgetting', label_font)
    plt.xlabel('task', label_font)
    for model, data in data_dict.items():
        data = np.array(data)
        xs = np.arange(2, data.shape[1] + 2)
        ys = data.mean(axis=0)
        plt.xticks(xs, size=16)
        plt.yticks(size=16)

        plt.plot(xs, ys, linestyle_dict[model], label=legend_dict[model], marker='o', markersize=3)
        fs = data.std(axis=0)
        plt.fill_between(xs, ys-fs, ys+fs, color=linestyle_dict[model][0], alpha=0.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.show()

def print_metric_table(data_dict):
    for model, data in data_dict.items():
        data = np.array(data)
        print('{}: mean = {:.4f} std = {:.4f}'.format(model, data.mean(axis=0)[-1], data.std(axis=0)[-1]), )


def show_feature_distrubtion(task_features, tasks, size=100, classes=3):
    labels_list = list(set(to_numpy(task_features[tasks[0]][1])))
    label_color_dict = {l:i for i, l in enumerate(labels_list)}
    for i, t in enumerate(tasks):
        fs, ls = task_features[t]
        features = []
        colors = []
        for l in labels_list[:classes]:
            features.append(fs[torch.nonzero(ls == l)[:size].squeeze()])
            colors += [label_color_dict[l]] * size
        features = to_numpy(torch.cat(features, dim=0))
        colors = np.array(colors)
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features = tsne.fit_transform(features)

        plt.subplot(2, 2, i+1)
        plt.scatter(features[:, 0], features[:, 1], c=colors, s=15, alpha=0.8)
        plt.xlim(-30, 30)
        plt.ylim(-30, 30)
    plt.show()

def show_task_distrubtion(task_features, tasks, size=500):
    features = []
    task_idxs = []
    labels_list = list(set(to_numpy(task_features[tasks[0]][1])))
    for i, t in enumerate(tasks):
        fs, ls = task_features[t]
        l = labels_list[1]
        idx = torch.nonzero(ls == l)[:size].squeeze()
        features.append(to_numpy(fs[idx]))
        task_idxs.append([t] * len(idx) )
    features = np.concatenate(features, axis=0)
    task_idxs = np.array(task_idxs).reshape(-1)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    features = tsne.fit_transform(features)
    scatter = plt.scatter(features[:,0], features[:,1], c=task_idxs, alpha=0.99)
    plt.legend(*scatter.legend_elements())
    plt.show()

def show_task_feature_distrubtion(task_features, tasks, size=100, classes=3):
    labels_list = list(set(to_numpy(task_features[tasks[0]][1])))
    label_color_dict = {l:i for i, l in enumerate(labels_list)}
    features = []
    colors = []
    for i, t in enumerate(tasks):
        fs, ls = task_features[t]
        t_features = []
        t_colors = []
        for l in labels_list[:classes]:
            idx = torch.nonzero(ls == l)[:size].squeeze()
            t_features.append(fs[idx])
            t_colors += [label_color_dict[l]] * len(idx)
        t_features = to_numpy(torch.cat(t_features, dim=0))
        t_colors = np.array(t_colors)
        features.append(t_features)
        colors.append(t_colors)
    features_ = np.concatenate(features, axis=0)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    features_ = tsne.fit_transform(features_)
    for i, t_colors in enumerate(colors):
        t_features = features_[i* len(t_colors): (i+1) * len(t_colors)]
        plt.subplot(2, 2, i+1)
        plt.xlim(-70, 70)
        plt.ylim(-60,60)
        plt.scatter(t_features[:, 0], t_features[:, 1], c=t_colors, s=15, alpha=0.8)

    plt.show()






if __name__ == '__main__':
    result_path = 'result_2'
    dataset = 'cifar100_10'
    # dataset = 'miniimageNet64_10'
    models = ['UCIR']
    seeds = [0,1,2,3,4 ]
    epoch = 6
    n_memories = 2000
    bn_types = ['bn', 'nbn']
    regular_types = ['None','decoder']
    model = models[0]
    legend_dict = {f'{model}+None+bn':f'{model}', f'{model}+decoder+bn': f'{model}_Decoder', f'{model}+None+nbn':f'{model}_NBN',
                   f'{model}+decoder+nbn':f'{model}_Decoder_NBN'}
    linestyle_dict = {f'{model}+None+bn':'b', f'{model}+decoder+bn': 'r', f'{model}+None+nbn':'g', f'{model}+decoder+nbn':'y'}
    model_acc_dict = defaultdict(list)
    model_fgt_dict = defaultdict(list)
    model_feature_dict = defaultdict(list)
    for model in models:
        for seed in seeds:
            for regular_type in regular_types:
                for bn_type in bn_types:
                    file_name = os.path.join(result_path, f'{model}_{dataset}_{seed}_{epoch}_{n_memories}_{bn_type}_{regular_type}.pt')
                    if not os.path.exists(file_name):
                        continue
                    result = torch.load(file_name)
                    task_accuracy = result.tasks_accuracy
                    first_task_feature = result.first_task_feature
                    acc, fgt = compute_average_accuracy(task_accuracy, epoch)
                    target = f'{model}+{regular_type}+{bn_type}'
                    model_acc_dict[target].append(acc)
                    model_fgt_dict[target].append(fgt)
                    model_feature_dict[target].append(first_task_feature)
    # plot_acc_lines(model_acc_dict, linestyle_dict, legend_dict)
    # plot_fgt_lines(model_fgt_dict, linestyle_dict, legend_dict)
    # print_metric_table(model_acc_dict)
    # show_feature_distrubtion(model_feature_dict['UCIR+None+nbn'][0], tasks=[0, 5])
    show_feature_distrubtion(model_feature_dict['UCIR+decoder+nbn'][0], tasks=[0, 2,4,6])

    # show_task_distrubtion(model_feature_dict['UCIR+decoder+bn'][1], tasks=[0, 1,2,3])
    # show_task_feature_distrubtion(model_feature_dict['UCIR+None+bn'][1], tasks=[0, 2,4,6])

