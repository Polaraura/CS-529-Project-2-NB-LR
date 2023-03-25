class LogisticRegressionHyperparameters:
    def __init__(self,
                 learning_rate,
                 penalty_term,
                 num_iter,
                 num_iter_print,
                 num_iter_save,
                 num_iter_validation,
                 validation_accuracy_diff_cutoff=0.05,
                 validation_split=0.2):
        """
        Make sure num_iter_save divides num_iter evenly

        Edit: actually, doesn't have to anymore

        :param learning_rate:
        :param penalty_term:
        :param num_iter:
        :param num_iter_print:
        :param num_iter_save:
        """

        self.learning_rate = learning_rate
        self.penalty_term = penalty_term
        self.num_iter = num_iter
        self.num_iter_print = num_iter_print
        self.num_iter_save = num_iter_save
        self.num_iter_validation = num_iter_validation
        self.validation_accuracy_diff_cutoff = validation_accuracy_diff_cutoff
        self.validation_split = validation_split
