import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import HypergraphConv
from layer import TemporalAttentionLayer

device = torch.device('cuda:0')


class THGLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim, num_window, with_dropout):
        super(THGLP, self).__init__()
        self.latent_dim = latent_dim

        self.conv_for_node = nn.ModuleList()
        self.conv_for_node.append(HypergraphConv(input_dim * 2, latent_dim[0] * 2))
        for i in range(1, len(latent_dim)):
            self.conv_for_node.append(HypergraphConv(latent_dim[i - 1] * 2, latent_dim[i] * 2))

        self.conv_for_edge = nn.ModuleList()
        self.conv_for_edge.append(HypergraphConv(input_dim, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_for_edge.append(HypergraphConv(latent_dim[i - 1], latent_dim[i]))

        self.collapsed_conv_for_node = nn.ModuleList()
        self.collapsed_conv_for_node.append(HypergraphConv(input_dim * 2, latent_dim[0] * 2))
        for i in range(1, len(latent_dim)):
            self.collapsed_conv_for_node.append(HypergraphConv(latent_dim[i - 1] * 2, latent_dim[i] * 2))

        self.collapsed_conv_for_edge = nn.ModuleList()
        self.collapsed_conv_for_edge.append(HypergraphConv(input_dim, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.collapsed_conv_for_edge.append(HypergraphConv(latent_dim[i - 1], latent_dim[i]))

        latent_dim = sum(latent_dim) * 4

        self.temporal_layer = TemporalAttentionLayer(latent_dim,
                                                     1,
                                                     num_window,
                                                     attn_drop=0.5,
                                                     residual=True)

        self.linear1 = nn.Linear(latent_dim * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)

        self.with_dropout = with_dropout

    def forward(self, graph_list):
        structural_outs = []
        for graph in graph_list[:-1]:
            g = copy.deepcopy(graph)
            g = g.to(device)
            x, hyperedge_index, edge_x, marks, edge_marks = g.x, g.edge_index, g.edge_attr, g.marks, g.edge_marks

            all_x = []
            all_edge_x = []
            lv = 0
            while lv < len(self.latent_dim):
                x1 = self.conv_for_node[lv](x, hyperedge_index)
                x1 = F.relu(x1)
                all_x.append(x1)

                edge_x1 = self.conv_for_edge[lv](edge_x, hyperedge_index[[1, 0]])
                edge_x1 = F.relu(edge_x1)
                all_edge_x.append(edge_x1)

                x = x1
                edge_x = edge_x1

                lv += 1

            x_out = torch.cat(all_x, 1)
            x_out = x_out[marks]
            edge_x_out = torch.cat(all_edge_x, 1)
            edge_x_out = torch.cat(
                [torch.min(edge_x_out[edge_marks], edge_x_out[edge_marks + 1]),
                 torch.max(edge_x_out[edge_marks], edge_x_out[edge_marks + 1])], 1)
            structural_out = torch.cat([edge_x_out, x_out], 1)
            structural_outs.append(structural_out)

        structural_outs = [x[:, None, :] for x in structural_outs]
        structural_outs = torch.cat(structural_outs, dim=1)

        temporal_outs = self.temporal_layer(structural_outs)[:, -1, :]

        collapsed_g = copy.deepcopy(graph_list[-1])
        collapsed_g = collapsed_g.to(device)
        x, hyperedge_index, edge_x, marks, edge_marks = collapsed_g.x, collapsed_g.edge_index, collapsed_g.edge_attr, collapsed_g.marks, collapsed_g.edge_marks

        collapsed_all_x = []
        collapsed_all_edge_x = []
        lv = 0
        while lv < len(self.latent_dim):
            x1 = self.collapsed_conv_for_node[lv](x, hyperedge_index)
            x1 = F.relu(x1)
            collapsed_all_x.append(x1)

            edge_x1 = self.collapsed_conv_for_edge[lv](edge_x, hyperedge_index[[1, 0]])
            edge_x1 = F.relu(edge_x1)
            collapsed_all_edge_x.append(edge_x1)

            x = x1
            edge_x = edge_x1

            lv += 1

        collapsed_x_out = torch.cat(collapsed_all_x, 1)
        collapsed_x_out = collapsed_x_out[marks]
        collapsed_edge_x_out = torch.cat(collapsed_all_edge_x, 1)
        collapsed_edge_x_out = torch.cat(
            [torch.min(collapsed_edge_x_out[edge_marks], collapsed_edge_x_out[edge_marks + 1]),
             torch.max(collapsed_edge_x_out[edge_marks], collapsed_edge_x_out[edge_marks + 1])], 1)
        collapsed_structural_out = torch.cat([collapsed_edge_x_out, collapsed_x_out], 1)

        output = torch.cat([temporal_outs, collapsed_structural_out], dim=1)

        output = self.linear1(output)
        output = F.relu(output)
        if self.with_dropout:
            output = F.dropout(output, training=self.training)
        output = self.linear2(output)
        output = F.log_softmax(output, dim=1)

        return output
