import torch
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import numpy as np
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F


torch.manual_seed(1111)
np.random.seed(1111)
EPS = 1e-13


def normalize(input):
    norm_square = (input ** 2).sum(dim=1)
    return input / torch.sqrt(norm_square.view(-1, 1))


class MultiInnerProductDecoder(torch.nn.Module):
    def __init__(self, in_dim, num_et):
        super(MultiInnerProductDecoder, self).__init__()
        self.num_et = num_et
        self.in_dim = in_dim
        self.weight = Parameter(torch.Tensor(num_et, in_dim))

        self.reset_parameters()

    def forward(self, z, edge_index, edge_type, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]] * self.weight[edge_type]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
        self.weight.data.normal_(std=1/np.sqrt(self.in_dim))


class myGCN(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 **kwargs):
        super(myGCN, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = np.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.fill_(0)

        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class PP(torch.nn.Module):

    def __init__(self, in_dim, nhid_list):
        super(PP, self).__init__()
        self.out_dim = nhid_list[-1]

        self.embedding = torch.nn.Parameter(torch.Tensor(in_dim, nhid_list[0]))

        self.conv_list = torch.nn.ModuleList(
            [myGCN(nhid_list[i], nhid_list[i + 1], cached=True) for i in range(len(nhid_list) - 1)]
        )


        # TODO:
        self.embedding.requires_grad = False

        # TODO:

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.data.normal_()

    def forward(self, x, pp_edge_index, edge_weight):
        tmp = []

        x = self.embedding

        # TODO
        # x = normalize(x)
        # TODO

        tmp.append(x)

        for net in self.conv_list[:-1]:
            x = net(x, pp_edge_index, edge_weight)

            # TODO
            # x = normalize(x)
            x = F.relu(x, inplace=True)
            # TODO

            tmp.append(x)

        x = self.conv_list[-1](x, pp_edge_index, edge_weight)


        # TODO
        # x = normalize(x)
        x = F.relu(x, inplace=True)
        # TODO

        tmp.append(x)

        # TODO
        # print([torch.abs(a).detach().mean().tolist() for a in tmp])
        # self.tmp = tmp
        # [a.retain_grad() for a in self.tmp]
        # TODO

        # TODO
        return torch.cat(tmp, dim=1)
        # TODO


class PD(torch.nn.Module):

    def __init__(self, protein_dim, d_dim_prot, n_drug, d_dim_feat=32):
        super(PD, self).__init__()
        self.p_dim = protein_dim
        self.d_dim_prot = d_dim_prot
        self.d_dim_feat = d_dim_feat
        self.n_drug = n_drug
        self.d_feat = torch.nn.Parameter(torch.Tensor(n_drug, d_dim_feat))

        # TODO:
        self.d_feat.requires_grad = False
        # TODO:

        self.conv = myGCN(protein_dim, d_dim_prot, cached=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.d_feat.data.normal_()

    def forward(self, x, pd_edge_index, edge_weight=None):

        n_prot = x.shape[0]
        tmp = pd_edge_index + 0
        tmp[1, :] += n_prot

        x = torch.cat([x, torch.zeros((self.n_drug, x.shape[1])).to(x.device)], dim=0)
        x = self.conv(x, tmp, edge_weight)[n_prot:, :]
        x = F.relu(x)
        x = torch.cat([x, torch.abs(self.d_feat)], dim=1)
        return x


class Model(torch.nn.Module):
    def __init__(self, pp, pd, mip):
        super(Model, self).__init__()
        self.pp = pp
        self.pd = pd
        self.mip = mip


class Pre_mask(torch.nn.Module):
    def __init__(self, pp_n_link, pd_n_link):
        super(Pre_mask, self).__init__()
        self.pp_weight = Parameter(torch.Tensor(pp_n_link))
        self.pd_weight = Parameter(torch.Tensor(pd_n_link))
        self.reset_parameters()

    def reset_parameters(self):
        # self.pp_weight.data.normal_(mean=0.5, std=0.01)
        # self.pd_weight.data.normal_(mean=0.5, std=0.01)
        self.pp_weight.data.fill_(0.99)
        self.pd_weight.data.fill_(0.99)

    # no change when at 0 or 1
    def desaturate(self):
        mask = self.pp_weight.data > 0.99
        self.pp_weight.data[mask] = 0.99

        mask = self.pp_weight.data < 0.01
        self.pp_weight.data[mask] = 0.01

        mask = self.pd_weight.data > 0.99
        self.pd_weight.data[mask] = 0.99

        mask = self.pd_weight.data < 0.01
        self.pd_weight.data[mask] = 0.01

    def saturate(self):
        mask = self.pp_weight.data >= 0.99
        self.pp_weight.data[mask] = 1.0

        mask = self.pp_weight.data <= 0.01
        self.pp_weight.data[mask] = 0.0

        mask = self.pd_weight.data >= 0.99
        self.pd_weight.data[mask] = 1.0

        mask = self.pd_weight.data <= 0.01
        self.pd_weight.data[mask] = 0.0


class Tip_explainer(object):
    def __init__(self, model, data, device):
        super(Tip_explainer, self).__init__()
        self.model = model
        self.data = data
        self.device = device

    def explain(self, drug_list_1, drug_list_2, side_effect_list, regulization=1):

        data = self.data
        model = self.model
        device = self.device

        pre_mask = Pre_mask(data.pp_index.shape[1] // 2, data.pd_index.shape[1]).to(device)
        data = data.to(device)
        model = model.to(device)

        for gcn in self.model.pp.conv_list:
            gcn.cached = False
        self.model.pd.conv.cached = False
        self.model.eval()

        pp_static_edge_weights = torch.ones((data.pp_index.shape[1])).to(device)
        pd_static_edge_weights = torch.ones((data.pd_index.shape[1])).to(device)

        optimizer = torch.optim.Adam(pre_mask.parameters(), lr=0.01)
        fake_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        z = model.pp(data.p_feat, data.pp_index, pp_static_edge_weights)
        z = model.pd(z, data.pd_index, pd_static_edge_weights)

        # P = torch.sigmoid((z[drug1, :] * z[drug2, :] * model.mip.weight[side_effect, :]).sum())
        P = torch.sigmoid((z[drug_list_1] * z[drug_list_2] * model.mip.weight[side_effect_list]).sum(dim=1))

        if len(drug_list_1) < 10:
            print(P.tolist())

        tmp = 0.0
        pre_mask.reset_parameters()
        for i in range(9999):
            model.train()
            pre_mask.desaturate()
            optimizer.zero_grad()
            fake_optimizer.zero_grad()

            # half_mask = torch.sigmoid(pre_mask.pp_weight)
            half_mask = torch.nn.Hardtanh(0, 1)(pre_mask.pp_weight)
            pp_mask = torch.cat([half_mask, half_mask])

            pd_mask = torch.nn.Hardtanh(0, 1)(pre_mask.pd_weight)

            z = model.pp(data.p_feat, data.pp_index, pp_mask)

            # TODO:
            # z = model.pd(z, data.pd_index, pd_static_edge_weights)
            z = model.pd(z, data.pd_index, pd_mask)
            # TODO:

            # P = torch.sigmoid((z[drug1, :] * z[drug2, :] * model.mip.weight[side_effect, :]).sum())
            P = torch.sigmoid((z[drug_list_1] * z[drug_list_2] * model.mip.weight[side_effect_list]).sum(dim=1))
            EPS = 1e-7

            # TODO:
            loss = torch.log(1 - P + EPS).sum() / regulization \
                   + 0.5 * (pp_mask * torch.pow((2 - pp_mask), 2)).sum() \
                   + (pd_mask * torch.pow((2 - pd_mask), 2)).sum()
            # loss = -  torch.log(P) + 0.5 * (pp_mask * (2 - pp_mask)).sum() + (pd_mask * (2 - pd_mask)).sum()
            # TODO:

            loss.backward()
            optimizer.step()
            # print("Epoch:{}, loss:{}, prob:{}, pp_link_sum:{}, pd_link_sum:{}".format(i, loss.tolist(), P.tolist(), pp_mask.sum().tolist(), pd_mask.sum().tolist()))

            print("Epoch:{:3d}, loss:{:0.2f}, prob:{:0.2f}, pp_link_sum:{:0.2f}, pd_link_sum:{:0.2f}".format(i, loss.tolist(), P.mean().tolist(), pp_mask.sum().tolist(), pd_mask.sum().tolist()))

            # until no weight need to be updated --> no sum of weights changes
            if tmp == pp_mask.sum().tolist() + pd_mask.sum().tolist():
                break
            else:
                tmp = pp_mask.sum().tolist() + pd_mask.sum().tolist()


        pre_mask.saturate()

        pp_left_mask = (pp_mask > 0.2).detach().cpu().numpy()
        tmp = (data.pp_index[0, :] > data.pp_index[1, :]).detach().cpu().numpy()
        pp_left_mask = np.logical_and(pp_left_mask, tmp)

        pd_left_mask = (pd_mask > 0.2).detach().cpu().numpy()

        pp_left_index = data.pp_index[:, pp_left_mask].cpu().numpy()
        pd_left_index = data.pd_index[:, pd_left_mask].cpu().numpy()

        pp_left_weight = pp_mask[pp_left_mask].detach().cpu().numpy()
        pd_left_weight = pd_mask[pd_left_mask].detach().cpu().numpy()

        return pp_left_index, pp_left_weight, pd_left_index, pd_left_weight

