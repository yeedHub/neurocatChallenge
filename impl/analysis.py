import numpy as np

from impl.fgsm_attack import FGSMAttack
from interface.analysis import ModelAnalysisInterface


class FunctionalityAnalysis(ModelAnalysisInterface):
    def __init__(self, num_class):
        self.__num_class = num_class  # number of different classes
        self.reset()

    def reset(self):
        self.__total_samples = 0  # number of all samples seen so far
        self.__total_class_samples = np.zeros(self.__num_class)  # number of each class in data seen so far

        self.__cnf_matrix = np.zeros((self.__num_class, self.__num_class))
        self.__TrueP = np.zeros(self.__num_class)
        self.__TrueN = np.zeros(self.__num_class)
        self.__FalseP = np.zeros(self.__num_class)
        self.__FalseN = np.zeros(self.__num_class)

        self.__recall = 0
        self.__specificity = 0
        self.__precision = 0

        self.__accuracy = 0
        self.__bal_accuracy = 0
        self.__weightedF1 = 0

    def __call__(self, model, input, params=None):
        X = input[0]
        Ytrue = input[1]
        for i, x in enumerate(X):
            interpreted_output = model.interpret_output(model.forward(x))

            self.count_samples(interpreted_output[0], Ytrue[i])

            self.calculate_confusion_matrix(interpreted_output[0], Ytrue[i])

        self.calculate_cnf_derivations()

        return {"total_samples": self.__total_samples,
                "total_class_samples": self.__total_class_samples,
                "cnf_matrix": self.__cnf_matrix,
                "TP": self.__TrueP,
                "TN": self.__TrueN,
                "FP": self.__FalseP,
                "FN": self.__FalseN,
                "accuracy": self.__accuracy,
                "balanced_accuracy": self.__bal_accuracy,
                "weightedF1": self.__weightedF1,
                }

    def count_samples(self, x, ytrue):
        unique, counts = np.unique(ytrue, return_counts=True)

        self.__total_class_samples[unique] += counts
        self.__total_samples += len(x)

    def calculate_confusion_matrix(self, output, ytrue):
        for i, pred in enumerate(output):
            if pred == ytrue[i]:
                self.__cnf_matrix[pred, pred] += 1
            else:
                self.__cnf_matrix[ytrue[i], pred] += 1

    def calculate_cnf_derivations(self):
        self.__TrueP = np.diag(self.__cnf_matrix)
        self.__FalseP = self.__cnf_matrix.sum(axis=0) - self.__TrueP
        self.__FalseN = self.__cnf_matrix.sum(axis=1) - self.__TrueP
        self.__TrueN = self.__cnf_matrix.sum() - (self.__TrueP + self.__FalseN + self.__FalseP)

        denominator = np.sum(self.__TrueN) + np.sum(self.__FalseP)
        self.__specificity = np.sum(self.__TrueN) / denominator if denominator != 0 else 1

        denominator = np.sum(self.__TrueP) + np.sum(self.__FalseP)
        self.__precision = np.sum(self.__TrueP) / denominator if denominator != 0 else 1

        denominator = np.sum(self.__TrueP) + np.sum(self.__FalseN)
        self.__recall = np.sum(self.__TrueP) / denominator if denominator != 0 else 1

        denominator = self.__precision + self.__recall
        F1 = 2 * (self.__precision * self.__recall) / denominator if denominator != 0 else 1

        self.__weightedF1 = np.sum(F1 * self.__total_class_samples) / self.__total_samples
        self.__bal_accuracy = np.sum(self.__recall + self.__specificity) / 2.0
        self.__accuracy = (np.sum(self.__TrueP) + np.sum(self.__TrueN)) / (
                np.sum(self.__TrueP) + np.sum(self.__TrueN) + np.sum(self.__FalseP) + np.sum(self.__FalseN))


class RobustnessAnalysis(ModelAnalysisInterface):
    def __init__(self, num_class):
        self.__func_analysis = FunctionalityAnalysis(num_class)

    def reset(self):
        self.__func_analysis.reset()

    def __call__(self, model, input, params=None):
        X = input[0]
        Ytrue = input[1]
        result = []
        attack = FGSMAttack()
        for i, x in enumerate(X):
            gradient = model.gradient_for(x, Ytrue[i])
            perturbed = None
            if params != None and "fgsm_eps" in params:
                perturbed = attack(params["fgsm_eps"], x, gradient)
            else:
                perturbed = attack(0.007, x, gradient)

            # If the prediction of original data is wrong, don't include them
            # (otherwise we run the risk of skewing our attack result)
            interpreted_output = model.interpret_output(model.forward(x))
            mask = interpreted_output[0] != Ytrue[i]
            perturbed[mask] = x[mask]

            result.append(perturbed)

        return self.__func_analysis(model, [result, Ytrue])
