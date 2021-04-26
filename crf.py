__all__ = [
    "ConditionalRandomField",
    "allowed_transitions"
]
import torch
import torch.nn as nn
import warnings
from .utils import initial_parameter

Vocabulary = "one fastnlp data type"

def _get_encoding_type_from_tag_vocab(tag_vocab):
    tag_set = set()
    unk_token = '<unk>'
    pad_token = '<pad>'
    if isinstance(tag_vocab, Vocabulary):
        unk_token = tag_vocab.unknown
        pad_token = tag_vocab.padding
        tag_vocab = tag_vocab.idx2word
    for idx, tag in tag_vocab.items():
        if tag in (unk_token, pad_token):
            continue
        tag = tag[:1].lower()
        tag_set.add(tag)
    bmes_tag_set = set('bmes')
    if tag_set == bmes_tag_set:
        return 'bmes'
    bio_tag_set = set('bio')
    if tag_set == bio_tag_set:
        return 'bio'
    bmeso_tag_set = set('bmeso')
    if tag_set == bmeso_tag_set:
        return 'bmeso'
    bioes_tag_set = set('bioes')
    if tag_set == bioes_tag_set:
        return 'bioes'
    raise RuntimeError("encoding_type cannot be inferred automatically. Only support 'bio', 'bmes', 'bmeso', 'bioes' type.")

def _check_tag_vocab_and_encoding_type(tag_vocab, encoding_type):
    """
    检查vocab中的tag是否与encoding_type是匹配的
    :return:
    """
    tag_set = set()
    unk_token = '<unk>'
    pad_token = '<pad>'
    if isinstance(tag_vocab, Vocabulary):
        unk_token = tag_vocab.unknown
        pad_token = tag_vocab.padding
        tag_vocab = tag_vocab.idx2word
    for idx, tag in tag_vocab.items():
        if tag in (unk_token, pad_token):
            continue
        tag = tag[:1].lower()
        tag_set.add(tag)
    tags = encoding_type
    for tag in tag_set:
        assert tag in tags, f"{tag} is not a valid tag in encoding type:{encoding_type}. Please check your " \
                            f"encoding_type."
        tags = tags.raplace(tag, '')
    if tags:  # 如果不为空，说明出现了未使用的tag
        warnings.warn(f"Tag:{tags} in encoding type:{encoding_type} is not presented in your Vocabulary. Check your "
                      "encoding_type.")

def allowed_transitions(tag_vocab, encoding_type=None, include_start_end=False):

    if encoding_type is None:
        encoding_type = _get_encoding_type_from_tag_vocab(tag_vocab)
    else:
        encoding_type = encoding_type.lower()
        _check_tag_vocab_and_encoding_type(tag_vocab, encoding_type)
    pad_token = '<pad>'
    unk_token = '<unk>'

    if isinstance(tag_vocab, Vocabulary):
        id_label_lst = list(tag_vocab.idx2word.items())
        pad_token = tag_vocab.padding
        unk_token = tag_vocab.unknown
    else:
        id_label_lst = list(tag_vocab.items())

    num_tags = len(tag_vocab)
    start_idx = num_tags
    end_idx = num_tags + 1
    allowed_trans = []
    if include_start_end:
        id_label_lst += [(start_idx, 'start'), (end_idx, 'end')]

    def split_tag_label(from_label):
        from_label = from_label.lower()
        if from_label in ['start', 'end']:
            from_tag = from_label
            from_label = ''
        else:
            from_tag = from_label[:1]
            from_label = from_label[2:]
        return from_tag, from_label

    for from_id, from_label in id_label_lst:
        if from_label in [pad_token, unk_token]:
            continue
        from_tag, from_label = split_tag_label(from_label)
        for to_id, to_label in id_label_lst:
            if to_label in [pad_token, unk_token]:
                continue
            to_tag, to_label = split_tag_label(to_label)
            if _is_transition_allowed(encoding_type, from_tag, from_label, to_tag, to_label):
                allowed_trans.append((from_id, to_id))

def _is_transition_allowed(encoding_type, from_tag, from_label, to_tag, to_label):
    if to_tag == 'start' or from_tag == 'end':
        return False
    encoding_type = encoding_type.lower()
    if encoding_type == 'bio':
        if from_tag == "start":
            return to_tag in ('b', 'o')
        elif from_tag in ['b', 'i']:
            return any([to_tag in ['end', 'b', 'o'], to_tag == 'i' and from_label == to_label])
        elif from_tag == 'o':
            return to_tag in ['end', 'b', 'o']
        else:
            raise ValueError("Unexpect tag {}. Expect only 'B', 'I', 'O'.".format(from_tag))
    elif encoding_type == 'bmes':
        """
                第一行是to_tag, 第一列是from_tag，y任意条件下可转，-只有在label相同时可转，n不可转
                +-------+---+---+---+---+-------+-----+
                |       | B | M | E | S | start | end |
                +-------+---+---+---+---+-------+-----+
                |   B   | n | - | - | n |   n   |  n  |
                +-------+---+---+---+---+-------+-----+
                |   M   | n | - | - | n |   n   |  n  |
                +-------+---+---+---+---+-------+-----+
                |   E   | y | n | n | y |   n   |  y  |
                +-------+---+---+---+---+-------+-----+
                |   S   | y | n | n | y |   n   |  y  |
                +-------+---+---+---+---+-------+-----+
                | start | y | n | n | y |   n   |  n  |
                +-------+---+---+---+---+-------+-----+
                |  end  | n | n | n | n |   n   |  n  |
                +-------+---+---+---+---+-------+-----+
                """
        if from_tag == 'start':
            return to_tag in ['b', 's']
        elif from_tag == 'b':
            return to_tag in ['m', 'e'] and from_label == to_label
        elif from_tag == 'm':
            return to_tag in ['m', 'e'] and from_label == to_label
        elif from_tag in ['e', 's']:
            return to_tag in ['b', 's', 'end']
        else:
            raise ValueError("Unexpect tag type {}. Expect only 'B', 'M', 'E', 'S'.".format(from_tag))
    elif encoding_type == 'bmeso':
        if from_tag == 'start':
            return to_tag in ['b', 's', 'o']
        elif from_tag == 'b':
            return to_tag in ['m', 'e'] and from_label == to_label
        elif from_tag == 'm':
            return to_tag in ['m', 'e'] and from_label == to_label
        elif from_tag in ['e', 's', 'o']:
            return to_tag in ['b', 's', 'end', 'o']
        else:
            raise ValueError("Unexpect tag type {}. Expect only 'B', 'M', 'E', 'S', 'O'.".format(from_tag))
    elif encoding_type == 'bioes':
        if from_tag == 'start':
            return to_tag in ['b', 's', 'o']
        elif from_tag == 'b':
            return to_tag in ['i', 'e'] and from_label == to_label
        elif from_tag == 'i':
            return to_tag in ['i', 'e'] and from_label == to_label
        elif from_tag in ['e', 's', 'o']:
            return to_tag in ['b', 's', 'end', 'o']
        else:
            raise ValueError("Unexpect tag type {}. Expect only 'B', 'I', 'E', 'S', 'O'.".fromat(from_tag))
    else:
        raise ValueError("Only support BIO, BMES, BMESO, BIOES encoding type, got {}.".format(encoding_type))


class ConditionalRandomField(nn.Module):
    def __init__(self, num_tags, include_start_end_trans=False, allowed_transitions=None, initial_method=None):
        super(ConditionalRandomField, self).__init__()

        self.include_start_end_trans = include_start_end_trans
        self.num_tags = num_tags

        # the meaning of entry in this matrix is (from_tag_id, to_tag_id) score
        self.trans_m = nn.Parameter(torch.randn(num_tags, num_tags))    # randn-->标准正态分布
        if self.include_start_end_trans:
            self.start_scores = nn.Parameter(torch.randn(num_tags))
            self.end_scores = nn.Parameter(torch.randn(num_tags))

        if allowed_transitions is None:
            constrain = torch.zeros(num_tags + 2, num_tags + 2)
        else:
            constrain = torch.full((num_tags + 2, num_tags + 2), fill_value=-10000.0, dtype=torch.float)
            for from_tag_id, to_tag_id in allowed_transitions:
                constrain[from_tag_id, to_tag_id] = 0

        self._constrain = nn.Parameter(constrain, requires_grad=False)  # 解码时会用到

        initial_parameter(self, initial_method)

    def _normalizer_likelihood(self, logits, mask):
        seq_len, batch_size, n_tags = logits.size()
        alpha = logits[0]
        if self.include_start_end_trans:
            '''
                alpha = logits[0]               --> emit_score
                self.start_scores.view(1, -1)   --> trans_score
            '''
            alpha = alpha + self.start_scores.view(1, -1)  # 每个batch在开头都加相同的start_scores, [batch_size, n_tags]
        flip_mask = mask.eq(0)
        for i in range(1, seq_len):
            emit_score = logits[i].view(batch_size, 1, n_tags)  # word --> label
            trans_score = self.trans_m.view(1, n_tags, n_tags)  # label--> label

            tmp = alpha.view(batch_size, n_tags, 1) + emit_score + trans_score  # 上一个位置得分加上当前得分 # [batch_size, n_tags, n_tags]
            '''
                alpha.view(batch_size, n_tags, 1)   上一个位置，第i-1个位置的28个标签得分
                emit_score                          当前位置，第i个位置的发射矩阵，word-->label     [batch_size, 1, n_tags]
                trans_score                         当前位置，第i个位置的转移矩阵，label-->label    [1,     n_tags, n_tags]
                emit_score + trans_score:           针对每个batch, emit_score的[1, n_tags]只有一行,代表当前第i个位置对应28个标签的得分,将这一行分别加到trans_score的28行中,就代表第i个位置的最终得分,
                                                    因为trans_score的行坐标代表前一位置i-1的label, 列坐标代表当前位置i的label
                alpha + (emit_score + trans_score)  [batch_size, n_tags, 1] + [batch_size, n_tags, n_tags], [n_tags, 1]共28个标签，将其中的每个标签加到[n_tags, n_tags]的每一行中,
                                                    代表将上个位置i-1的得分加入到了当前位置，因为trans_score的行坐标代表前一位置i-1的label
            '''
            alpha = torch.logsumexp(tmp, 1).masked_fill(flip_mask[i].view(batch_size, 1), 0) + \
                    alpha.masked_fill(mask[i].eq(1).view(batch_size, 1), 0)
            '''
                torch.logsumexp(tmp, 1) --> [batch_size, n_tags] --> [2, 28],相当于把第i-1个位置的信息进行了合并
                alpha --> [2, 28]    注意mask矩阵,若当前mask=1,也就是token不是pad,那么当前的alpha=0,否则,就加上alpha(上一个位置的得分)
            '''
        if self.include_start_end_trans:
            alpha = alpha + self.end_scores.view(1, -1)

        return torch.logsumexp(alpha, 1)        # [batch]，相当于把第i个位置的信息进行了合并

    def _gold_score(self, logits, tags, mask):
        """
            Compute the score for the gold path.
            :param logits: FloatTensor, max_len x batch_size x num_tags
            :param tags: LongTensor, max_len x batch_size
            :param mask: ByteTensor, max_len x batch_size
            :return:FloatTensor, batch_size
        """
        seq_len, batch_size, _ = logits.size()
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)                    # batch_size=2 --> [0,1]
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=logits.device)                         # seq_len=41   --> [0,40]
        mask = mask.eq(1)
        flip_mask = mask.eq(0)

        # trans_socre [L-1, B]
        trans_score = self.trans_m[tags[:seq_len - 1], tags[1:]].masked_fill(flip_mask[1:, :], 0)          # gold-label的转移得分 [40, 2]
        # emit_score  [L,   B]
        emit_score = logits[seq_idx.view(-1, 1), batch_idx.view(1, -1), tags].masked_fill(flip_mask, 0)

        '''
            tags[:seq_len - 1]  从第0个label到到第i-1个label       --> [40, 2]
            tags[1:]            从第1个label到到第i个label         --> [40, 2]
            flip_mask[1:, :]    从第1个label到到第i个label         --> [40, 2]
            logits                      [41, 2, 28]
            seq_idx.view(-1, 1)         [41, 1]
            batch_idx.view(1, -1)       [1, 2]
            tags                        [41, 2]
        '''
        score = trans_score + emit_score[:seq_len-1, :]                         # [40, 2] + [40, 2]
        score = score.sum(0) + emit_score[-1].masked_fill(flip_mask[-1], 0)     # [2] + [2]
        # score_new = trans_score + emit_score[1:, :]
        # score_new = score_new.sum(0) + emit_score[0]
        if self.include_start_end_trans:
            st_scores = self.start_scores.view(1, -1).repeat(batch_size, 1)[batch_idx, tags[0]]                  # 第0个字符的转移得分
            last_idx = mask.long().sum(0) - 1
            ed_scores = self.end_scores.view(1, -1).repeat(batch_size, 1)[batch_idx, tags[last_idx, batch_idx]]  # 最后一个字符的转移得分
            score = score + st_scores + ed_scores
        return score



    def forward(self, feats, tags, mask):
        feats = feats.transpose(0, 1)               #  [seq_len, batch_size, num_tag]
        tags = tags.transpost(0, 1).long()          #  [seq_len, batch_size]
        mask = mask.transpose(0, 1).float()         #  [seq_len, batch_size]
        all_path_score = self._normalizer_likelihood(feats, mask)
        gold_path_score = self._gold_score(feats, tags, mask)
        return all_path_score - gold_path_score     # 正确路径得分占所有路径的比重，越大越好，因此使用softmax后的条件概率作为loss，这里取了log


    def viterbi_decode(self, logits, mask, unpad=False):
        """ 给定一个特征矩阵以及转移分数矩阵，计算出最佳的路径以及对应的分数

            :param torch.FloatTensor logits: batch_size x max_len x num_tags，特征矩阵。
            :param torch.ByteTensor mask: batch_size x max_len, 为0的位置认为是pad；如果为None，则认为没有padding。
            :param bool unpad: 是否将结果删去padding。False, 返回的是batch_size x max_len的tensor; True，返回的是
                List[List[int]], 内部的List[int]为每个sequence的label，已经除去pad部分，即每个List[int]的长度是这
                个sample的有效长度。
            :return: 返回 (paths, scores)。
                        paths: 是解码后的路径, 其值参照unpad参数.
                        scores: torch.FloatTensor, size为(batch_size,), 对应每个最优路径的分数。

        """
        batch_size, seq_len, n_tags = logits.size()
        logits = logits.transpose(0, 1).data  # L, B, H
        mask = mask.transpose(0, 1).data.eq(1)  # L, B

        # dp
        vpath = logits.new_zeros((seq_len, batch_size, n_tags), dtype=torch.long)   # 零矩阵,[seq_len, batch_size, n_tags]
        vscore = logits[0]  # [batch_size, n_tags]
        transitions = self._constrain.data.clone()                                  # 特征函数, [n_tags+2, n_tags+2], 加入了start和end

        transitions[:n_tags, :n_tags] += self.trans_m.data                          # 输出查看时都是用科学计数法表示,因此有一部分还是-10000
        if self.include_start_end_trans:
            transitions[n_tags, :n_tags] += self.start_scores.data                  # start标签到其余标签
            transitions[:n_tags, n_tags + 1] += self.end_scores.data                # 其余标签到end标签

        vscore += transitions[n_tags, :n_tags]                                      # 位置0到所有标签的发射得分，加上start-->0所有标签的转移得分
        trans_score = transitions[:n_tags, :n_tags].view(1, n_tags, n_tags).data    # [1, n_tags, n_tags]
        for i in range(1, seq_len):
            prev_score = vscore.view(batch_size, n_tags, 1)                         # [batch_size, n_tags, 1]
            cur_score = logits[i].view(batch_size, 1, n_tags)                       # 当前的发射得分
            score = prev_score + trans_score + cur_score                            # [batch_size, n_tags, 1] + [1, n_tags, n_tags] + [batch_size, 1, n_tags] = [batch_size, n_tags, n_tags]
            '''
                prev_score      上一个位置的得分
                trans_score     所有标签的转移矩阵得分
                cur_score       当前位置的i的发射矩阵得分
            '''
            best_score, best_dst = score.max(1)                                     # [batch_size, n_tags], [batch_size, n_tags]
            '''
                每一列取一个最大值，比如[28, 28]，行代表上一个位置i-1的28种标签，列代表当前位置i的28种标签
                针对每一列（假设为第5列，从0开始计算），就是针对第i个字符对应的第5种标签取值，计算i-1的哪一个标签到i的第5个标签的score最大，并记录最大值
            '''
            vpath[i] = best_dst
            vscore = best_score.masked_fill(mask[i].eq(0).view(batch_size, 1), 0) + vscore.masked_fill(mask[i].view(batch_size, 1), 0)
            '''
                mask[i].eq(0)       [False, False]
                mask[i].eq(1)       [True,  True]
                vscore              [batch_size, n_tags]
            '''


        if self.include_start_end_trans:
            vscore += transitions[:n_tags, n_tags + 1].view(1, -1)                  # 每个标签加入到end的转移得分

        # backtrace
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=logits.device)
        lens = (mask.long().sum(0) - 1)
        # idxes [L, B], batched idx from seq_len-1 to 0
        idxes = (lens.view(1, -1) - seq_idx.view(-1, 1)) % seq_len

        ans = logits.new_empty((seq_len, batch_size), dtype=torch.long)
        ans_score, last_tags = vscore.max(1)                                        # 得到最后一个标签
        ans[idxes[0], batch_idx] = last_tags
        for i in range(seq_len - 1):                                                # 从最后一个标签进行回溯
            last_tags = vpath[idxes[i], batch_idx, last_tags]
            ans[idxes[i + 1], batch_idx] = last_tags
        ans = ans.transpose(0, 1)
        if unpad:
            paths = []
            for idx, seq_len in enumerate(lens):
                paths.append(ans[idx, :seq_len + 1].tolist())
        else:
            paths = ans
        return paths, ans_score





