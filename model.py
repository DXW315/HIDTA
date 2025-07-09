import torch
import torch.nn as nn
import torch.nn.functional as F
import util


class GateGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GateGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gate = nn.Sigmoid()  # 门控机制

    def forward(self, x):
        h, _ = self.gru(x)
        gate_output = self.gate(h)
        return gate_output


class PathChoiceDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PathChoiceDecoder, self).__init__()
        self.gate_gru = GateGRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)  # 输出路径选择的成本

    def forward(self, x):
        gate_output = self.gate_gru(x)
        path_cost = self.fc(gate_output)  # 计算路径选择成本
        return path_cost


class SequentialConsistencyLoss(nn.Module):
    def __init__(self, lambda_consistency=0.1):
        super(SequentialConsistencyLoss, self).__init__()
        self.lambda_consistency = lambda_consistency  # 控制一致性正则化的强度

    def forward(self, predictions, real_values):
        # Calculate temporal differences
        time_diff = torch.abs(predictions[:, :, 1:] - predictions[:, :, :-1])  # |pred_t - pred_(t-1)|
        real_diff = torch.abs(real_values[:, :, 1:] - real_values[:, :, :-1])  # |real_t - real_(t-1)|

        # Consistency loss, penalize large changes
        consistency_loss = torch.mean(time_diff / (real_diff + 1e-6))  # 防止除以零
        return self.lambda_consistency * consistency_loss


class HIDTA(nn.Module):
    def __init__(self, batch_size, H_a, H_b, G0, G1, indices, G0_all, G1_all, H_T_new, lwjl, num_nodes,
                 dropout=0.3, supports=None, in_dim=1, out_dim=12, residual_channels=40, dilation_channels=40,
                 skip_channels=320, end_channels=640, kernel_size=2, blocks=3, layers=1):
        super(HIDTA, self).__init__()
        self.batch_size = batch_size
        self.H_a = H_a
        self.H_b = H_b
        self.G0 = G0
        self.G1 = G1
        self.H_T_new = H_T_new
        self.lwjl = lwjl
        self.indices = indices
        self.G0_all = G0_all
        self.G1_all = G1_all

        # Initialize parameters
        self.edge_node_vec1 = nn.Parameter(torch.rand(self.H_a.size(1), 10).cuda(), requires_grad=True).cuda()
        self.edge_node_vec2 = nn.Parameter(torch.rand(10, self.H_a.size(0)).cuda(), requires_grad=True).cuda()

        self.node_edge_vec1 = nn.Parameter(torch.rand(self.H_a.size(0), 10).cuda(), requires_grad=True).cuda()
        self.node_edge_vec2 = nn.Parameter(torch.rand(10, self.H_a.size(1)).cuda(), requires_grad=True).cuda()

        self.hgcn_w_vec_edge_At_forward = nn.Parameter(torch.rand(self.H_a.size(1)).cuda(), requires_grad=True).cuda()
        self.hgcn_w_vec_edge_At_backward = nn.Parameter(torch.rand(self.H_a.size(1)).cuda(), requires_grad=True).cuda()

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        # Define the convolutional laye
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.dgconv = nn.ModuleList()
        self.filter_convs_h = nn.ModuleList()
        self.gate_convs_h = nn.ModuleList()
        self.SAt_forward = nn.ModuleList()
        self.SAt_backward = nn.ModuleList()
        self.hgconv_edge_At_forward = nn.ModuleList()
        self.hgconv_edge_At_backward = nn.ModuleList()
        self.gconv_dgcn_w = nn.ModuleList()
        self.dhgconv = nn.ModuleList()
        self.bn_g = nn.ModuleList()
        self.bn_hg = nn.ModuleList()

        # Introduce the PathChoiceDecoder
        self.path_choice_decoder = PathChoiceDecoder(input_size=residual_channels, hidden_size=64, output_size=num_nodes)

        # Initialize other parameters
        self.bn_start = nn.BatchNorm2d(in_dim, affine=False)
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))

        self.supports = supports
        self.num_nodes = num_nodes
        receptive_field = 1
        self.supports_len = 0
        self.supports_len += len(supports)
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).cuda(), requires_grad=True).cuda()
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).cuda(), requires_grad=True).cuda()
        self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size
            new_dilation = 2
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)

        self.receptive_field = receptive_field

        self.new_supports_w = [
            torch.zeros([self.num_nodes, self.num_nodes]).repeat([self.batch_size, 1, 1]).cuda()
        ]
        self.new_supports_w = self.new_supports_w + [
            torch.zeros([self.num_nodes, self.num_nodes]).repeat([self.batch_size, 1, 1]).cuda()
        ]
        self.new_supports_w = self.new_supports_w + [
            torch.zeros([self.num_nodes, self.num_nodes]).repeat([self.batch_size, 1, 1]).cuda()
        ]

    def forward(self, input):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x = self.bn_start(x)
        x = self.start_conv(x)

        skip = 0
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        adp_new = adp.repeat([self.batch_size, 1, 1])
        new_supports = self.supports + [adp]

        edge_node_H = (self.H_T_new * (torch.mm(self.edge_node_vec1, self.edge_node_vec2)))
        self.H_a_ = (self.H_a * (torch.mm(self.node_edge_vec1, self.node_edge_vec2)))
        self.H_b_ = (self.H_b * (torch.mm(self.node_edge_vec1, self.node_edge_vec2)))
        G0G1_edge_At_forward = self.G0_all @ (torch.diag_embed((self.hgcn_w_vec_edge_At_forward))) @ self.G1_all
        G0G1_edge_At_backward = self.G0_all @ (torch.diag_embed((self.hgcn_w_vec_edge_At_backward))) @ self.G1_all

        self.new_supports_w[2] = adp_new.cuda()
        forward_medium = torch.eye(self.num_nodes).repeat([self.batch_size, 1, 1]).cuda()
        backward_medium = torch.eye(self.num_nodes).repeat([self.batch_size, 1, 1]).cuda()

        for i in range(self.blocks * self.layers):
            edge_feature = util.feature_node_to_edge(x, self.H_a_, self.H_b_, operation="concat")
            edge_feature = torch.cat([edge_feature, self.lwjl.repeat(1, 1, 1, edge_feature.size(3))], dim=1)

            filter_h = self.filter_convs_h[i](edge_feature)
            filter_h = torch.tanh(filter_h)
            gate_h = self.gate_convs_h[i](edge_feature)
            gate_h = torch.sigmoid(gate_h)
            x_h = filter_h * gate_h

            batch_edge_forward = self.SAt_forward[i](x.transpose(1, 2), self.indices[0], self.indices[1])
            batch_edge_backward = self.SAt_backward[i](x.transpose(1, 2), self.indices[0], self.indices[1])

            batch_edge_forward = torch.unsqueeze(batch_edge_forward, dim=3).transpose(1, 2)
            batch_edge_forward = self.hgconv_edge_At_forward[i](batch_edge_forward, G0G1_edge_At_forward)
            batch_edge_forward = torch.squeeze(batch_edge_forward)
            forward_medium[:, self.indices[0], self.indices[1]] = torch.sigmoid(batch_edge_forward)
            self.new_supports_w[0] = forward_medium

            batch_edge_backward = torch.unsqueeze(batch_edge_backward, dim=3).transpose(1, 2)
            batch_edge_backward = self.hgconv_edge_At_backward[i](batch_edge_backward, G0G1_edge_At_backward)
            batch_edge_backward = torch.squeeze(batch_edge_backward)
            backward_medium[:, self.indices[0], self.indices[1]] = torch.sigmoid(batch_edge_backward)
            self.new_supports_w[1] = backward_medium.transpose(1, 2)

            self.new_supports_w[0] = self.new_supports_w[0] * new_supports[0]
            self.new_supports_w[1] = self.new_supports_w[1] * new_supports[1]

            residual = x

            filter_ = self.filter_convs[i](residual)
            filter_ = torch.tanh(filter_)
            gate_ = self.gate_convs[i](residual)
            gate_ = torch.sigmoid(gate_)
            x = filter_ * gate_
            x = self.dgconv[i](x, self.new_supports_w)
            x = self.bn_g[i](x)

            dhgcn_w_input = residual
            dhgcn_w_input = dhgcn_w_input.transpose(1, 2)
            dhgcn_w_input = torch.mean(dhgcn_w_input, 3)
            dhgcn_w_input = dhgcn_w_input.transpose(0, 2)
            dhgcn_w_input = torch.unsqueeze(dhgcn_w_input, dim=0)
            dhgcn_w_input = self.gconv_dgcn_w[i](dhgcn_w_input, self.supports)
            dhgcn_w_input = torch.squeeze(dhgcn_w_input)
            dhgcn_w_input = dhgcn_w_input.transpose(0, 1)
            dhgcn_w_input = self.G0 @ (torch.diag_embed(dhgcn_w_input)) @ self.G1

            x_h = self.dhgconv[i](x_h, dhgcn_w_input)
            x_h = self.bn_hg[i](x_h)

            x = util.fusion_edge_node(x, x_h, edge_node_H)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

        x = F.leaky_relu(skip)
        x = F.leaky_relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
