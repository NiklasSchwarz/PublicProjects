import tensorflow as tf

class Pix2PixScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, decay_start, total_steps, end_lr=0.0, power=1.0, name=None):
        super().__init__()
        self.init_lr = float(init_lr)
        self.decay_start = int(decay_start)
        self.total_steps = int(total_steps)
        self.end_lr = float(end_lr)
        self.power = float(power)
        self.name = name or "Pix2PixScheduler"

        self._poly = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.init_lr,
            decay_steps=self.total_steps - self.decay_start,
            end_learning_rate=self.end_lr,
            power=self.power
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # bis decay_start: konstante LR; danach: PolynomialDecay mit verschobenem Step
        return tf.where(step < self.decay_start,
                        tf.constant(self.init_lr, tf.float32),
                        self._poly(step - self.decay_start))

    def get_config(self):
        return {
            "init_lr": self.init_lr,
            "decay_start": self.decay_start,
            "total_steps": self.total_steps,
            "end_lr": self.end_lr,
            "power": self.power,
            "name": self.name,
        }