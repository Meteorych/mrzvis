# Лабораторная работа 2 по дисциплине МРЗвИС
# Выполнена студентом группы 121703 БГУИР Титлов И.Д.
# Реализовать модель сети Хопфилда с непрерывным состоянием и дискретным временем в асинхронном режиме.
# Вариант 3

# Ссылки на источники:
# https://numpy.org/doc/2.1/reference/index.html
# https://matplotlib.org/stable/api/index
# https://habr.com/ru/articles/301406
# https://github.com/Meteorych/mrzvis

import numpy as np
import matplotlib.pyplot as plt


def preprocess_alphabet(alphabet):
    processed_alphabet = []
    for item in alphabet:
        if isinstance(item, np.ndarray) and item.ndim == 2:
            processed_alphabet.append(item.ravel())
        else:
            processed_alphabet.append(item)
    return np.array(processed_alphabet)


class Hopfield:
    def __init__(self, images, nu: float = 1.0) -> None:
        self.size = images.shape[1]
        self.w = np.zeros((self.size, self.size))
        self.images = images
        self.neg_images = self._get_neg_images(self.images)
        self.nu = nu

    def _get_neg_images(self, images):
        return images * -1

    def train(self, e=1e-6, max_iters=10000):
        for _ in range(max_iters):
            old_w = self.w.copy()

            for image in self.images:
                x_t = np.matrix(image.copy()).T
                activation = np.tanh(self.w @ x_t)
                self.w += (self.nu / self.size) * (x_t - activation) @ x_t.T
                np.fill_diagonal(
                    self.w, 0
                )  # Диагональные элементы обнуляются, чтобы нейроны не влияли сами на себя.

            # Условие сходимости
            if np.abs(old_w - self.w).sum() < e:
                print(f"Количество итераций обучения: {_}")
                break

        np.fill_diagonal(self.w, 0)

    def _find_image_num(self, x, images) -> int | None:
        for idx, image in enumerate(images):
            if np.abs(image - x).max() < 1e-2:
                return idx
        return None

    def predict(self, x, max_iters: int = 1000):
        states = [np.matrix(x.copy())] * 4
        relaxation_iters = 0

        for _ in range(max_iters):
            relaxation_iters += 1

            # Calculate the new state
            new_state = np.tanh(self.w @ states[-1].T).T
            states.append(new_state)
            states.pop(0)

            # Check for convergence
            if (
                _ >= 3
                and np.abs(states[0] - states[2]).max() < 1e-8
                and np.abs(states[1] - states[3]).max() < 1e-8
            ):
                # Find the closest image from the training set
                image_num = self._find_image_num(new_state, self.images)
                neg_image_num = self._find_image_num(new_state, self.neg_images)
                is_negative = neg_image_num is not None

                return (
                    relaxation_iters,
                    new_state,
                    (image_num if image_num is not None else neg_image_num),
                    is_negative,
                )

        return max_iters, new_state, None, None


def display_console_image_result(original, result):
    plt.imshow(original.reshape((4, 4)))
    plt.show()

    print(result)

    plt.imshow(result.reshape((4, 4)))
    plt.show()


alphabet = np.array(
    [
        [[-1, 1, 1, -1], [1, -1, -1, 1], [1, 1, 1, 1], [1, -1, -1, 1]],
        [[1, 1, 1, -1], [1, -1, -1, 1], [1, -1, -1, 1], [1, 1, 1, -1]],
        [[-1, 1, 1, -1], [1, -1, -1, 1], [1, -1, -1, 1], [-1, 1, 1, -1]],
        [[-1, 1, 1, 1], [1, -1, -1, -1], [1, -1, -1, -1], [-1, 1, 1, 1]],
    ]
)

# [-1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1],
# [1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1],
# [-1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
# # [-1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1],

# [-1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1]

fig, axes = plt.subplots(1, len(alphabet), figsize=(12, 3))  # 1 row, N columns

# Plot each matrix in a subplot
for i, ax in enumerate(axes):
    ax.imshow(
        alphabet[i].reshape((4, 4)), cmap="binary"
    )  # Use 'binary' colormap for binary matrices
    ax.set_title(f"Symbol {i+1}")
    ax.axis("off")  # Hide axes for cleaner visualization

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

alphabet = preprocess_alphabet(alphabet)
network = Hopfield(alphabet, 0.7)
network.train()
input_image = np.array([1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1])
iterations, predicted_state, _, _ = network.predict(input_image)

print("Iterations: ", iterations)

display_console_image_result(input_image, predicted_state)


# for i in range(self.size):
#         for j in range(self.size):
#             if i != j:  # Исключаем диагональные элементы
#                 activation = np.tanh(
#                     self.w[i, :] @ x_t
#                 )  # Скалярное произведение
#                 self.w[i, j] += (
#                     (self.nu / self.size)
#                     * (x_t[j, 0] - activation[0, 0])
#                     * x_t[i, 0]
#                 )

#     np.fill_diagonal(
#         self.w, 0
#     )  # Диагональные элементы обнуляются, чтобы нейроны не влияли сами на себя.

# if np.linalg.norm(self.w - old_w) < e:
#     break
