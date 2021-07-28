class LossLayer:
    def __init__(self, loss_function, loss_gradient):
        self.prediction = None
        self.desired_prediction = None
        self.loss_function = loss_function
        self.loss_gradient = loss_gradient
        self.error = 0

    def get_loss(self, prediction, desired_prediction):
        self.prediction = prediction
        self.desired_prediction = desired_prediction
        self.error = self.loss_function(prediction, desired_prediction)
        return self.error

    def get_gradient(self):
        return self.loss_gradient(self.prediction, self.desired_prediction)