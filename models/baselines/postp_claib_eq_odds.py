import numpy as np
from collections import namedtuple
from models.losses import get_binary_ocean_values
import torch

# adapted from https://github.com/gpleiss/equalized_odds_and_calibration
class PostModel():

    def __init__(self, pred, label, target_personality):
        self.target_personality = target_personality
        self.binary_pred = get_binary_ocean_values(pred, STE=False, target_personality=target_personality).float()
        # if tensor, convert to numpy
        self.pred = pred
        self.label = label
        self.eps = 1e-6
        if isinstance(self.pred, torch.Tensor):
            self.pred = self.pred.cpu()
            if len(self.pred.shape) > 1:
                self.pred = self.pred.squeeze(dim=1)

            self.label = self.label.cpu()
            if len(self.label.shape) > 1:
                self.label = self.label.squeeze(dim=1)

            self.pred = self.pred.numpy()
            self.label = self.label.numpy()

    def logits(self):
        raw_logits = np.clip(np.log(self.pred / (1 - self.pred)), -100, 100)
        return raw_logits

    def num_samples(self):
        return len(self.pred)

    def base_rate(self):
        """
        Percentage of samples belonging to the positive class
        """
        return np.mean(self.label)


    # mse are calculated outside
    # def mse(self):
    #     return np.mean(np.square(self.pred - self.label))

    def tpr(self):
        """
        True positive rate
        """
        return np.mean(np.logical_and(self.binary_pred == 1, self.label == 1))

    def tpr(self):
        """
        False positive rate
        """
        return np.mean(np.logical_and(self.binary_pred == 1, self.label == 0))

    def tnr(self):
        """
        True negative rate
        """
        return np.mean(np.logical_and(self.binary_pred == 0, self.label == 0))

    def tnr(self):
        """
        False negative rate
        """
        return np.mean(np.logical_and(self.binary_pred == 0, self.label == 1))



    def tp_cost(self):
        """
        Generalized true positive cost
        """
        if len(self.pred[self.label == 1]) == 0:
            return 0
        return self.pred[self.label == 1].mean()

    def tn_cost(self):
        """
        Generalized true negative cost
        """
        return 1 - self.pred[self.label == 0].mean()


    def calib_eq_odds(self, other, tp_rate, tn_rate, mix_rates=None):
        if tn_rate == 0:

            self_cost = self.tp_cost()
            other_cost = other.tp_cost()
            print(self_cost, other_cost)
            self_trivial_cost = self.trivial().tp_cost()
            other_trivial_cost = other.trivial().tp_cost()
        elif tp_rate == 0:
            self_cost = self.tn_cost()
            other_cost = other.tn_cost()
            self_trivial_cost = self.trivial().tn_cost()
            other_trivial_cost = other.trivial().tn_cost()
        else:
            self_cost = self.weighted_cost(tp_rate, tn_rate)
            other_cost = other.weighted_cost(tp_rate, tn_rate)
            self_trivial_cost = self.trivial().weighted_cost(tp_rate, tn_rate)
            other_trivial_cost = other.trivial().weighted_cost(tp_rate, tn_rate)

        other_costs_more = other_cost > self_cost
        self_mix_rate = (other_cost - self_cost + self.eps) / (self_trivial_cost - self_cost + self.eps) if other_costs_more else 0
        # other_mix_rate can be NAN
        # print(f"self_cost: {self_cost}, other_cost: {other_cost}, self_trivial_cost: {self_trivial_cost}, other_trivial_cost: {other_trivial_cost}")
        # print(f"{(self_cost - other_cost + self.eps)} , {other_trivial_cost - other_cost + self.eps}   ")
        other_mix_rate = 0 if other_costs_more else (self_cost - other_cost + self.eps) / (other_trivial_cost - other_cost + self.eps)

        # New classifiers
        self_indices = np.random.permutation(len(self.pred))[:int(self_mix_rate * len(self.pred))]
        self_new_pred = self.pred.copy()
        self_new_pred[self_indices] = self.base_rate()
        calib_eq_odds_self = PostModel(self_new_pred, self.label, self.target_personality)
        # print(f"self mix rate: {self_mix_rate}, other mix rate: {other_mix_rate}")
        # print(f"other mix rate: {other_mix_rate}, len other: {len(other.pred)}")
        # print( f"----------------other mix rate * len other.pred {other_mix_rate * len(other.pred)}")
        other_indices = np.random.permutation(len(other.pred))[:int(other_mix_rate * len(other.pred))]
        other_new_pred = other.pred.copy()
        other_new_pred[other_indices] = other.base_rate()
        calib_eq_odds_other = PostModel(other_new_pred, other.label, self.target_personality)

        if mix_rates is None:
            return calib_eq_odds_self, calib_eq_odds_other, (self_mix_rate, other_mix_rate)
        else:
            return calib_eq_odds_self, calib_eq_odds_other

    def trivial(self):
        """
        Given a classifier, produces the trivial classifier
        (i.e. a PostModel that just returns the base rate for every prediction)
        """
        base_rate = self.base_rate()
        pred = np.ones(len(self.pred)) * base_rate
        return PostModel(pred, self.label, self.target_personality)

    def weighted_cost(self, tp_rate, tn_rate):
        """
        Returns the weighted cost
        If tp_rate = 1 and tn_rate = 0, returns self.tp_cost
        If tp_rate = 0 and tn_rate = 1, returns self.tn_cost
        If tp_rate and tn_rate are nonzero, returns tp_rate * self.tp_cost * (1 - self.base_rate) +
            tn_rate * self.tn_cost * self.base_rate
        """
        norm_const = float(tp_rate + tn_rate) if (tp_rate != 0 and tn_rate != 0) else 1
        res = tp_rate / norm_const * self.tp_cost() * (1 - self.base_rate()) + \
            tn_rate / norm_const * self.tn_cost() * self.base_rate()
        return res

    # def __repr__(self):
    #     return '\n'.join([
    #         'mse:\t%.3f' % self.mse(),
    #         'F.P. cost:\t%.3f' % self.tp_cost(),
    #         'F.N. cost:\t%.3f' % self.tn_cost(),
    #         'Base rate:\t%.3f' % self.base_rate(),
    #
    #     ])

