# Лабораторная работа 3 по дисциплине МРЗвИС
# Выполнена студентом группы 121703 БГУИР Титлов И.Д.
#
# Вариант 3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import utils


if __name__ == "__main__":
    first_val = 0
    last_val = 100
    last_step = 1000
    steps_num = 10
    num_of_digits = 10
    alpha = 0.8
    func_of_activation = "linear"

    graphic = utils.print_message()
    range_of_succession = np.linspace(first_val, last_val, last_step)
    set = []
    for i in range_of_succession:
        tempList = []
        tempList.append(graphic(i))
        set.append(tempList)
    print(set)

    step = int(alpha * len(set))
    training_set = set[:step]
    testing_set = set[step:]
    train_main_set = training_set[utils.succession_len :]
    test_main_set = range_of_succession[step:]

    list_for_training = tf.keras.utils.timeseries_dataset_from_array(
        sequence_length=utils.succession_len,
        targets=train_main_set,
        batch_size=num_of_digits,
        data=training_set,
    )

    LSTM = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(utils.succession_len, 1)),
            tf.keras.layers.LSTM(
                steps_num, activation=func_of_activation, return_sequences=False
            ),
            tf.keras.layers.Dense(units=1),
        ]
    )

    LSTM.compile(
        loss=tf.losses.Huber(),
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        metrics=[tf.metrics.MeanAbsoluteError()],
    )

    LSTM.build()
    LSTM.summary()

    utils.train_lstm(LSTM, list_for_training, steps_num)
    forecast = utils.test_lstm(LSTM, testing_set)

    plt.plot(test_main_set, testing_set)
    plt.plot(test_main_set, forecast)
    plt.show()
