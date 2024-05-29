import torch
import torch.nn as nn
import torch.nn.functional as F

# reproduction of 'Mitigating Unwanted Biases with Adversarial Learning'
# adapted from https://github.com/giandos200/Zhang-et-al-Mitigating-Unwanted-Biases-with-Adversarial-Learning


class adversary_model(nn.Module):
    def __init__(self):
        super(adversary_model, self).__init__()
        self.c = torch.FloatTensor([1.0]) # requires_grad=True
        # self.c.requires_grad = True

        # inputs are s, s * true_labels, s * (1.0 - true_labels); output is one personality
        self.FC1 = nn.Linear(3, 1)
        self.sigmoid = torch.sigmoid


    def forward(self,pred_logits, true_labels):

        s = self.sigmoid((1+torch.abs(self.c.to(pred_logits.device))) * pred_logits)
        pred_protected_attribute_logits = self.FC1(torch.cat([s, s * true_labels, s * (1.0 - true_labels)], 1))
        pred_protected_attribute_labels = self.sigmoid(pred_protected_attribute_logits)
        return pred_protected_attribute_labels, pred_protected_attribute_logits