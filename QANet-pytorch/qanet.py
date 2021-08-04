import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from args import get_train_args

args = get_train_args()

D = args.d_model
Nh = args.num_heads
Dword = args.glove_dim
Dchar = args.char_dim
batch_size = args.batch_size
dropout = args.dropout
dropout_char = args.dropout_char

Lc = args.para_limit
Lq = args.ques_limit


def mask_logits(inputs, mask):
    mask = mask.type(torch.float32)
    return inputs + (-1e30) * (1 - mask)


class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, relu=False, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.out = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                             bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu == True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)  # [batch_size, hidden_size, seq_len] -> [batch_size, seq_len, hidden_size]
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)  # [1, seq_len, hidden_size]
    return (x + signal.cuda()).transpose(1, 2)  # [batch_size, hidden_size, seq_len]  # TODO


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)  # [length, num_timescales]
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  # [length, channels]
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                        padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class Highway(nn.Module):
    def __init__(self, layer_num: int, size=D):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([Initialized_Conv1d(size, size, relu=False, bias=True) for _ in range(self.n)])
        self.gate = nn.ModuleList([Initialized_Conv1d(size, size, bias=True) for _ in range(self.n)])

    def forward(self, x):
        # x: shape [batch_size, hidden_size, length]
        for i in range(self.n):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
            # x = F.relu(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.mem_conv = Initialized_Conv1d(D, D * 2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(D, D, kernel_size=1, relu=False, bias=False)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        memory = queries

        memory = self.mem_conv(memory)  # [batch_size, 2*D, seq_len]
        query = self.query_conv(queries)  # [batch_size, D, seq_len]
        memory = memory.transpose(1, 2)  # [batch_size, seq_len, 2*D]
        query = query.transpose(1, 2)  # [batch_size, seq_len, D]
        Q = self.split_last_dim(query, Nh)  # [batch_size, heads, seq_len, D//heads]
        K, V = [self.split_last_dim(tensor, Nh) for tensor in torch.split(memory, D, dim=2)]

        key_depth_per_head = D // Nh
        Q *= key_depth_per_head ** -0.5
        x = self.dot_product_attention(Q, K, V, mask=mask)  # [batch_size, heads, seq_len, D//heads]
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3)).transpose(1, 2)  # [batch_size, D, seq_len]

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))  # [batch, heads, length_q, length_kv]
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])  # [batch, 1, 1, length]
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=dropout, training=self.training)
        return torch.matmul(weights, v)  # [batch, heads, length_q, depth_v]

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(Dchar, D, kernel_size=(1, 5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(Dword + D, D, bias=False)
        self.high = Highway(2)

    def forward(self, ch_emb, wd_emb):
        N = ch_emb.size()[0]
        # wd_emb [batch_size, seq_len, Dword]
        # ch_emb [batch_size, seq_len, char_limit ,char_dim]
        ch_emb = ch_emb.permute(0, 3, 1, 2)  # [batch_size, char_dim, seq_len, char_limit]
        ch_emb = F.dropout(ch_emb, p=dropout_char, training=self.training)
        ch_emb = self.conv2d(ch_emb)  # [batch_size, D, seq_len, char_limit-5+1]
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)
        ch_emb = ch_emb.squeeze()  # [batch_size, D, seq_len]

        wd_emb = F.dropout(wd_emb, p=dropout, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)  # [batch_size, Dword, seq_len]
        emb = torch.cat([ch_emb, wd_emb], dim=1)  # [batch_size, Dword+D, seq_len]
        emb = self.conv1d(emb)  # [batch_size, D, seq_len]
        emb = self.high(emb)
        return emb


class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention()
        self.FFN_1 = Initialized_Conv1d(ch_num, ch_num, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(ch_num, ch_num, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(D) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(D)
        self.norm_2 = nn.LayerNorm(D)
        self.conv_num = conv_num

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num + 1) * blks
        # x [batch_size, hidden_size, seq_len]
        # mask [batch_size, seq_len]
        # out [batch_size, hidden_size, seq_len]
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1, 2)).transpose(1, 2)
            if i % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
            l += 1
        res = out
        out = self.norm_1(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
        l += 1
        res = out

        out = self.norm_2(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class CQAttention(nn.Module):
    def __init__(self):
        super().__init__()
        w4C = torch.empty(D, 1)
        w4Q = torch.empty(D, 1)
        w4mlu = torch.empty(1, 1, D)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, C, Q, Cmask, Qmask):
        # ----------------------------------------------------------------------------------------
        # TODO
        # 补全代码 1
        # 使用下面的提示代码和课上的课件，补全下面 CQAttention 的计算过程
        # ----------------------------------------------------------------------------------------
        # C [batch_size, hidden_size, lc]
        # Q [batch_size, hidden_size, lq]
        C = C.transpose(1, 2)  # [batch_size, lc, hidden_size]
        Q = Q.transpose(1, 2)  # [batch_size, lq, hidden_size]
        batch_size_c = C.size(0)
        S = self.trilinear_for_attention(C, Q)  # [batch_size, lc, lq]
        Cmask = Cmask.view(batch_size_c, C.size(1), 1)
        Qmask = Qmask.view(batch_size_c, 1, Q.size(1))

        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)

        a = torch.bmm(S1, Q)  # [batch_size, lc, hidden_size]
        b = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)  # [batch_size, lc, hidden_size]

        out = torch.cat([C, a, torch.mul(C, a), torch.mul(C, b)], dim=2)  # [batch_size, lc, hidden_size*4]
        return out.transpose(1, 2)  # [batch_size, hidden_size*4, lc]

    def trilinear_for_attention(self, C, Q):
        C = F.dropout(C, p=dropout, training=self.training)  # [batch_size, lc, hidden_size]
        Q = F.dropout(Q, p=dropout, training=self.training)  # [batch_size, lq, hidden_size]
        lc = C.size(1)
        lq = Q.size(1)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, lq])  # [batch_size, lc, lq]
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, lc, -1])  # [batch_size, lc, lq]
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))  # [batch_size, lc, lq]
        res = subres0 + subres1 + subres2
        res += self.bias
        return res  # [batch_size, lc, lq]


class Pointer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = Initialized_Conv1d(D * 2, 1)
        self.w2 = Initialized_Conv1d(D * 2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)
        Y1 = F.log_softmax(Y1, dim=1)
        Y2 = F.log_softmax(Y2, dim=1)
        return Y1, Y2


class QANet(nn.Module):
    def __init__(self, word_mat, char_mat):
        super().__init__()
        self.char_emb = nn.Embedding.from_pretrained(char_mat, freeze=args.pretrained_char)
        self.word_emb = nn.Embedding.from_pretrained(word_mat, freeze=True)
        self.emb = Embedding()
        self.emb_enc = EncoderBlock(conv_num=4, ch_num=D, k=7)
        self.cq_att = CQAttention()
        self.cq_resizer = Initialized_Conv1d(D * 4, D)
        self.model_enc_blks = nn.ModuleList([EncoderBlock(conv_num=2, ch_num=D, k=5) for _ in range(7)])
        self.out = Pointer()

    def forward(self, Cwid, Qwid, Ccid, Qcid):
        maskC = (torch.zeros_like(Cwid) != Cwid).float()
        maskQ = (torch.zeros_like(Qwid) != Qwid).float()
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)
        Ce = self.emb_enc(C, maskC, 1, 1)
        Qe = self.emb_enc(Q, maskQ, 1, 1)
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer(X)
        # ----------------------------------------------------------------------------------------
        # TODO
        # 补全代码 2
        # 补全下面代码，补全模型的输出
        # 计算 M1、M2、M3 和 p1、p2
        # ----------------------------------------------------------------------------------------
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M2 = M0
        M0 = F.dropout(M0, p=dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M3 = M0

        p1, p2 = self.out(M1, M2, M3, maskC)

        return p1, p2
