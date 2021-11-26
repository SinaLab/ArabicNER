import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_labels, start_label, stop_label):
        super().__init__()
        self.start_label = start_label
        self.stop_label = stop_label
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))

        # No transitions from any tag to start tag
        self.transitions.data[:, start_label] = 0

        # No transitions from stop tag to any tag
        self.transitions.data[stop_label, :] = 0

    def decode(self, emissions):
        """
        Decode the sequence given emissions/logits
        :param emissions: torch.Tensor - B x T x num_labels tensor, output of BERT
        :return:
        """

        backpointers = list()
        batch_size, seq_len, num_labels = emissions.shape

        # viterbivars is the vector that passes the viterbi variables/scores
        # as we move from one timestep to the next. At each timestep we add the
        # transition scores and emissions to viterbivars
        viterbivars = torch.full((1, num_labels), -10000.).to(emissions.device)
        viterbivars[:, self.start_label] = 0

        for i in range(1, seq_len):
            backpointers_t = list()
            viterbivars_t = list()

            for j in range(num_labels):
                t_score = viterbivars + self.transitions[:, j]
                best_label_score, best_label = torch.max(t_score, dim=-1)
                backpointers_t.append(best_label)
                viterbivars_t.append(best_label_score)

            backpointers.append(backpointers_t)
            viterbivars = torch.stack(viterbivars_t).t() + emissions[:, i]

        viterbivars += self.transitions[:, self.stop_label]
        best_final_score, best_final_label = torch.max(viterbivars, dim=-1)


    def forward(self, emissions):
        self.decode(emissions)
        pass
