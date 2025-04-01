import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import torch.nn.functional as F
from model import THGLP
from utils import fix_seed, loader, split_data, get_test_data, get_train_graph, links_to_subgraphs, to_hypergraphs

device = torch.device('cuda:0')

if __name__ == '__main__':
    # , 'dblp', 'as733', 'fbw', 'HepPh30'
    for data_name in ['enron10']:
        f_out = open('./' + data_name + '3.txt', 'w')
        mean_best_auc = []
        mean_best_ap = []
        for seed in range(3124, 3134):
            print(seed)
            latent_dim = [32, 32, 32, 1]
            hidden_size = 128
            hop = 2
            batch = 50
            test_ratio = 0
            lr = 0.0001
            num_epochs = 200
            early_stop = 20
            is_new = False
            is_multi = True
            dropout = True

            fix_seed(seed)
            data, test_len, trainable_feat = loader(data_name)

            # print(data.keys())
            # for i in range(len(data['edge_index_list'])):
            #     print(data['edge_index_list'][i].shape,
            #           data['pedges'][i].shape,
            #           data['nedges'][i].shape,
            #           data['new_pedges'][i].shape,
            #           data['new_nedges'][i].shape)
            # print(data['num_nodes'])
            # print(data['time_length'])
            # print(data['weights'])

            num_nodes = data['num_nodes']
            time_steps = data['time_length']

            train_shots = list(range(0, time_steps - test_len))
            test_shots = list(range(time_steps - test_len, time_steps))

            train_graph_list = get_train_graph(data, train_shots)

            train_pos, train_neg = split_data(data,
                                              train_shots[-1],
                                              test_ratio,
                                              is_new)

            test_pos, test_neg = get_test_data(data, test_shots, is_new, is_multi)

            train_subgraphs_list, test_subgraphs_list, max_num_node_labels = links_to_subgraphs(
                train_graph_list,
                train_pos,
                train_neg,
                test_pos,
                test_neg,
                hop)

            train_hypergraphs_list = to_hypergraphs(train_subgraphs_list, max_num_node_labels, batch, True)
            test_hypergraphs_list = to_hypergraphs(test_subgraphs_list, max_num_node_labels, batch, False)

            model = THGLP(int(max_num_node_labels + 1), hidden_size, latent_dim, len(train_shots), dropout).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            best_auc = 0
            best_ap = 0
            stop = 0
            for epoch in range(num_epochs):
                total_loss = []
                all_targets = []
                all_scores = []

                model.train()
                for hypergraph_list in train_hypergraphs_list:
                    y = hypergraph_list[0].y.to(device)
                    out = model(hypergraph_list)
                    loss = F.nll_loss(out, y)
                    total_loss.append(loss.data.cpu().detach())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                for hypergraph_list in test_hypergraphs_list:
                    y = hypergraph_list[0].y
                    out = model(hypergraph_list)
                    all_targets.extend(y.tolist())
                    all_scores.append(out[:, 1].cpu().detach())

                total_loss = np.array(total_loss)
                all_targets = np.array(all_targets)
                all_scores = torch.cat(all_scores).cpu().numpy()
                ap = metrics.average_precision_score(all_targets, all_scores)
                fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
                auc = metrics.auc(fpr, tpr)

                # print(('average test of epoch %d: loss %.5f auc %.5f ap %.5f' % (
                #     epoch, float(np.mean(total_loss)), auc, ap)))

                if auc > best_auc:
                    best_auc = auc
                    best_ap = ap
                    stop = 0
                else:
                    stop += 1
                    if stop > early_stop:
                        break
            print("{:.4f}\t{:.4f}".format(best_auc, best_ap))
            print("{:.4f}\t{:.4f}".format(best_auc, best_ap), file=f_out)
            mean_best_auc.append(best_auc)
            mean_best_ap.append(best_ap)
        print("{:.4f}\t{:.4f}".format(np.mean(mean_best_auc), np.mean(mean_best_ap)), file=f_out)
