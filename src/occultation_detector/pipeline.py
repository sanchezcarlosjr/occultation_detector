import dataclass
import tensorflow as tf

class WrongInput(Exception):
    def __init__(self, message="The model needs 92 timestamps to work"):
        self.message = message
        super().__init__(self.message)


def pipeline(path):
    model = Model(path)
    model.load_model()
    def predict(*dfs):
        df = pd.concat([df['intensity'].T for df in dfs])
        if len(intensity) != 92:
            raise WrongInput()
        return model.predict(df, verbose=0)
    return predict

class Model:
    def __init__(self, path="checkpoints/vanilla-neuronal-network.keras"):
        self.path = path
    def load_model(self):
        self.model = tf.keras.models.load_model(self.path)
    def predict(lightcurve):
        return self.model.predict(lightcurve)
