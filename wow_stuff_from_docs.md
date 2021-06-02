# Useful stuff from docs #

*Ha. Turns out reading documentation is helpful, wow.*

## Read dataset from directory without dumb loops ##

tf.keras.preprocessing.image_dataset_from_directory 
Turns image files sorted into class-specific folders into a labeled dataset of image tensors.

## any shape input!!! ##

To build models with the Functional API, you start by specifying the shape (and optionally the dtype) of your inputs. If any dimension of your input can vary, you can specify it as None. For instance, an input for 200x200 RGB image would have shape (200, 200, 3), but an input for RGB images of any size would have shape (None, None, 3).

*Gonna use for image classification with hand made data*

## Using callbacks for checkpointing (and more) ##

If training goes on for more than a few minutes, it's important to save your model at regular intervals during training. You can then use your saved models to restart training in case your training process crashes (this is important for multi-worker distributed training, since with many workers at least one of them is bound to fail at some point).

*WHATAFUCK so i neednt have waited again every fucking time when the shit crashed???*

You can use callbacks to periodically save your model. Here's a simple example: a ModelCheckpoint callback configured to save the model at the end of every epoch. The filename will include the current epoch.

~~~
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='path/to/my/model_{epoch}',
        save_freq='epoch')
]
model.fit(dataset, epochs=2, callbacks=callbacks)
~~~

*I can also send ame a message when training is complete or stream a training to a bot or do with training progress whatever i want*

## Damn, i can monnitor what happpens while training in a little web app##

TensorBoard, a web application that can display real-time graphs of your metrics (and more).

To use TensorBoard with fit(), simply pass a keras.callbacks.TensorBoard callback specifying the directory where to store TensorBoard logs:

~~~
callbacks = [
    keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(dataset, epochs=2, callbacks=callbacks)
~~~

## If i need to modify training loop eg gan or anything like that i can do it so ##

This shit happens in a built-in fit function, si i can write my own

~~~
class CustomModel(keras.Model):
  def train_step(self, data):
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    x, y = data
    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)  # Forward pass
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        loss = self.compiled_loss(y, y_pred,
                                    regularization_losses=self.losses)
    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # Update metrics (includes the metric that tracks the loss)
    self.compiled_metrics.update_state(y, y_pred)
    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}

# Construct and compile an instance of CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=[...])

# Just use `fit` as usual
model.fit(dataset, epochs=3, callbacks=...)
~~~


## To see what happens indide fit shit while trainings ##

~~~
model.compile(optimizer='adam', loss='mse', run_eagerly=True)
~~~

*run_eagerly=True is like a dubug mode which slows down training and gives u acsess to the insides*

TURN OFF WHEN COMPILE A MODEL (otherwise u'll fucking die untill it's over)

## somehow keras can tell me what the fuck do i do wrong ##

Once you have a working model, you're going to want to optimize its configuration -- architecture choices, layer sizes, etc. Human intuition can only go so far, so you'll want to leverage a systematic approach: hyperparameter search.

First, place your model definition in a function, that takes a single hp argument. Inside this function, replace any value you want to tune with a call to hyperparameter sampling methods, e.g. hp.Int() or hp.Choice():

~~~
def build_model(hp):
    inputs = keras.Input(shape=(784,))
    x = layers.Dense(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        activation='relu'))(inputs)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model
The function should return a compiled model.
~~~

Next, instantiate a tuner object specifying your optimization objective and other search parameters:

~~~
import kerastuner

tuner = kerastuner.tuners.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=100,
    max_trials=200,
    executions_per_trial=2,
    directory='my_dir')
~~~

Finally, start the search with the search() method, which takes the same arguments as Model.fit():

~~~
tuner.search(dataset, validation_data=val_dataset)
~~~

When search is over, you can retrieve the best model(s):

~~~
models = tuner.get_best_models(num_models=2)
~~~

Or print a summary of the results:

~~~
tun-er.results_summary()
~~~

*HA so i did it right when difining a special function for every fucking thing. *
*FUNCTIONAL PROGRAMING RULES*

## moooooore callbacks ##

some callbacks:

~~~
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
model.fit(dataset, epochs=10, callbacks=my_callbacks)
~~~

