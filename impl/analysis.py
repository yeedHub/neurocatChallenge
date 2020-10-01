import numpy as np

from impl.pytorchnn import PyTorchNNMultiClassifier
from interface.analysis import ModelAnalysisInterface
from interface.attack import AttackInterface


class FunctionalityAnalysis(ModelAnalysisInterface):
    """
    Implementation of a confusion matrix based functionality analysis.
    Calculates the confusion matrix and derives the following metrics from it:
    total number of samples seen,
    total number of class instances in seen samples,
    true positives,
    true negatives,
    false positives,
    false negatives,
    specificity,
    precision,
    recall,
    standard binary accuracy (extended to the multi-class case),
    balanced accuracy,
    weighted F1
    """

    def __init__(self, num_class: int) -> None:
        """

        Args:
            num_class: number of classes present in the dataset

        Returns: None
        """
        super().__init__()
        self.__num_class = num_class  # number of different classes
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

    def reset(self) -> None:
        """
        Reset the metrics calculated/collected while executing the analysis.

        Returns: None

        """
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

    def __call__(self, model: PyTorchNNMultiClassifier, model_input: tuple, params: dict = None) -> dict:
        """
        Execute the analysis.

        Args:
            model:
            model_input:
            params:

        Returns: a dictionary that contains all calculated metrics

        """
        samples = model_input[0]
        labels = model_input[1]
        for i, sample in enumerate(samples):
            interpreted_output = model.interpret_output(model.forward(sample))

            self.count_samples(interpreted_output[0], labels[i])

            self.calculate_confusion_matrix(interpreted_output[0], labels[i])

        self.calculate_cnf_derivations()

        return {"total_samples": self.__total_samples,
                "total_class_samples": self.__total_class_samples,
                "cnf_matrix": self.__cnf_matrix,
                "TP": self.__TrueP,
                "TN": self.__TrueN,
                "FP": self.__FalseP,
                "FN": self.__FalseN,
                "specificity": self.__specificity,
                "precision": self.__precision,
                "recall": self.__recall,
                "accuracy": self.__accuracy,
                "balanced_accuracy": self.__bal_accuracy,
                "weightedF1": self.__weightedF1,
                }

    def count_samples(self, sample: np.ndarray, label: np.ndarray) -> None:
        """
        Counts the number of samples and the class distribution thereof that the analysis has seen so far.

        Args:
            sample: a numpy n dimensional array containing the sample
            label: a numpy n dimensional array containing the label

        Returns: None

        """
        unique, counts = np.unique(label, return_counts=True)

        self.__total_class_samples[unique] += counts
        self.__total_samples += len(sample)

    def calculate_confusion_matrix(self, predictions: np.ndarray, label: np.ndarray) -> None:
        """
        Calculates the confusion matrix.
        Args:
            predictions: the interpreted output (or prediction) of the model for a given input
            label: the true label of the given input

        Returns: None

        """
        for i, prediction in enumerate(predictions):
            if prediction == label[i]:
                self.__cnf_matrix[prediction, prediction] += 1
            else:
                self.__cnf_matrix[label[i], prediction] += 1

    def calculate_cnf_derivations(self) -> None:
        """
        Calculate derivations from the confusion matrix.

        Returns: None

        """
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
        f1 = 2 * (self.__precision * self.__recall) / denominator if denominator != 0 else 1

        self.__weightedF1 = np.sum(f1 * self.__total_class_samples) / self.__total_samples
        self.__bal_accuracy = np.sum(self.__recall + self.__specificity) / 2.0
        self.__accuracy = (np.sum(self.__TrueP) + np.sum(self.__TrueN)) / (
                np.sum(self.__TrueP) + np.sum(self.__TrueN) + np.sum(self.__FalseP) + np.sum(self.__FalseN))


class RobustnessAnalysis(ModelAnalysisInterface):
    """
    Implementation of an attack based robustness analysis.
    """

    def __init__(self, attack: AttackInterface, num_class: int) -> None:
        """

        Args:
            attack: the attack to be executed
            num_class: number of classes present in the dataset

        Returns: None

        """
        super().__init__()
        self.__attack = attack
        self.__func_analysis = FunctionalityAnalysis(num_class)

    def reset(self):
        """
        Reset the metrics calculated/collected while executing the analysis.

        Returns: None

        """
        self.__func_analysis.reset()

    def __call__(self, model: PyTorchNNMultiClassifier, model_input: tuple) -> dict:
        """
        Execute the analysis.

        Args:
            model: the model on which the attack should be executed
            model_input: input to the model

        Returns: dict containing results of a functional analysis after the attack

        """
        samples = model_input[0]
        labels = model_input[1]
        result = []
        for i, sample in enumerate(samples):
            gradient = model.gradient_for((sample, labels[i]))
            perturbed = self.__attack((sample, gradient))

            # If the prediction of original data is wrong, don't include them
            # (otherwise we run the risk of skewing our attack result)
            interpreted_output = model.interpret_output(model.forward(sample))
            mask = interpreted_output[0] != labels[i]
            perturbed[mask] = sample[mask]

            result.append(perturbed)

        return self.__func_analysis(model, (result, labels))

    def set_attack_params(self, params: dict) -> None:
        """
        Set the parameters of the attack.

        Args:
            params: new parameters of the attack

        Returns: None

        """
        self.__attack.set_params(params)
