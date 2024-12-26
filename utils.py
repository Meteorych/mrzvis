from tqdm import tqdm
from math import sin, cos, tan

succession_len = 50
succession = {
    "sin(x)": lambda x: sin(x),
    "cos(x)": lambda x: cos(x),
    "tan(x)": lambda x: tan(x),
}


def print_message():
    while True:
        first_step = input("Введите функцию: ")
        succession1 = first_step.strip().lower()
        if succession.get(succession1) is not None:
            graphic = succession.get(succession1)
            break
        else:
            print("Я не знаю такой функции(")
    return graphic


def train_lstm(lstm, set, steps):
    for _ in range(steps):
        lstm.fit(set, validation_data=None)


def test_lstm(lstm, succession):
    test_output = []
    i = 0
    while i < succession_len:
        test_output.append(succession[i][0])
        i += 1
    first_point = [succession[:succession_len]]
    iteration = len(succession) - succession_len
    for _ in tqdm(range(iteration)):
        temp_point = lstm.predict(first_point, verbose=0)[0][0]
        next_point = float(temp_point)
        first_point[0] = first_point[0][1:]
        first_point[0].append([next_point])
        test_output.append(next_point)

    return test_output
