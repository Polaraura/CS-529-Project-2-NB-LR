class LogisticRegressionHyperparameters:
    def __init__(self,
                 learning_rate,
                 penalty_term,
                 num_iter):
        self.learning_rate = learning_rate
        self.penalty_term = penalty_term
        self.num_iter = num_iter
