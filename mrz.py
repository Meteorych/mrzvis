# Лабораторная работа 1 по дисциплине МРЗвИС
# Выполнена студентом группы 121703 БГУИР Титлов И.Д.
# Реализация линейной рециркуляционной сети с постоянным коэффициентом обучения с нормированными весами
# как модели самокодировщика для задачи понижения размерности данных
# Вариант 3

# Ссылки на источники:
# https://numpy.org/doc/2.1/reference/index.html
# https://matplotlib.org/stable/api/index
# https://habr.com/ru/articles/130581/

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2


class ImageCompressor:
    HIDDEN_SIZE = 42
    MAX_ERROR = 1500.0
    LEARNING_RATE = 0.00092

    @classmethod
    def split_into_blocks(cls, height, width):
        """
        Splits the image into blocks of size (BLOCK_HEIGHT, BLOCK_WIDTH)
        and stores them in a list.

        Args:
            height (int): The height of the image.
            width (int): The width of the image.

        Returns:
            np.ndarray: A 3D array of shape (n_blocks, BLOCK_HEIGHT, BLOCK_WIDTH, 3).
        """
        img = cls.load_image()
        blocks = []
        for i in range(height // cls.BLOCK_HEIGHT):
            for j in range(width // cls.BLOCK_WIDTH):
                block = img[
                    cls.BLOCK_HEIGHT * i : cls.BLOCK_HEIGHT * (i + 1),
                    cls.BLOCK_WIDTH * j : cls.BLOCK_HEIGHT * (j + 1),
                    :3,
                ]
                blocks.append(block)
        return np.array(blocks)

    @classmethod
    def blocks_to_image_array(
        cls, blocks: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        """
        Converts blocks of image data back into a 3D image array.

        Args:
            blocks (np.ndarray): A 3D array of shape (n_blocks, BLOCK_HEIGHT, BLOCK_WIDTH, 3).
            height (int): The height of the image.
            width (int): The width of the image.

        Returns:
            np.ndarray: A 3D array of shape (height, width, 3) representing the image.
        """
        image_array: list = []
        blocks_in_line: int = width // cls.BLOCK_WIDTH
        for i in range(height // cls.BLOCK_HEIGHT):
            for y in range(cls.BLOCK_HEIGHT):
                line: list = [
                    [
                        blocks[
                            i * blocks_in_line + j,
                            (y * cls.BLOCK_WIDTH * 3) + (x * 3) + color,
                        ]
                        for color in range(3)
                    ]
                    for j in range(blocks_in_line)
                    for x in range(cls.BLOCK_WIDTH)
                ]
                image_array.append(line)
        return np.array(image_array)

    @classmethod
    def display_image(cls, img_array: np.ndarray) -> None:
        """
        Displays the given image array.

        The image array is scaled to the range [0, 1] and then displayed using matplotlib.
        """
        scaled_image: np.ndarray = 1.0 * (img_array + 1) / 2
        # Turn off axis
        plt.axis("off")
        # Display the image
        plt.imshow(scaled_image)
        # Show the plot
        plt.show(block=False)

    @classmethod
    def load_image(cls):
        """
        Loads an image from a file and preprocesses it.

        The image is read from the file "home.bmp" in the current directory and
        converted to a floating-point array with values in the range [0, 1].
        The image is then converted from BGR to RGB color space and scaled to
        the range [-1, 1].

        Returns:
            np.ndarray: A 3D array of shape (height, width, 3) representing the
                preprocessed image.
        """
        image = (
            cv2.imread("desert.bmp", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return (2.0 * image) - 1.0

    @classmethod
    def get_image_dimensions(cls):
        """
        Returns the dimensions of the loaded image.

        Returns:
            tuple: A tuple (height, width) containing the dimensions of the image.
        """
        img = cls.load_image()
        return img.shape[0], img.shape[1]

    @classmethod
    def initialize_layers(cls):
        """
        Initializes the two layers of the linear autoencoder.

        The first layer is a matrix of shape (input_size, hidden_size) and the
        second layer is a matrix of shape (hidden_size, input_size). The layers
        are initialized randomly and then normalized.

        The layers are normalized by dividing each column of the matrix by its
        Euclidean norm (the square root of the sum of the squares of its
        elements). This ensures that the layers are initialized with a certain
        level of sparsity.

        Returns:
            tuple: A tuple (layer1, layer2) containing the two initialized and
                normalized layers.
        """
        layer1 = tf.random.uniform((cls.INPUT_SIZE, cls.HIDDEN_SIZE)) * 2 - 1

        layer2 = tf.transpose(layer1)

        layer1 = cls.normalize(layer1)

        layer2 = cls.normalize(layer2)

        return layer1, layer2

    @classmethod
    def normalize(cls, weight):
        """
        Normalizes the columns of a weight matrix.

        This method takes a TensorFlow weight matrix, converts it to a NumPy array,
        and normalizes each column of the matrix by its Euclidean norm (magnitude).
        If a column's magnitude is zero, its elements are set to zero to prevent
        division by zero. The normalized weight matrix is converted back to a
        TensorFlow tensor before being returned.

        Args:
            weight (tf.Tensor): A 2D TensorFlow tensor representing the weight matrix
                to be normalized.

        Returns:
            tf.Tensor: A 2D TensorFlow tensor with the same shape as the input,
            where each column has been normalized by its Euclidean norm.
        """
        weight = weight.numpy()
        denominator = cls.mod_of_vector(weight)

        for column_index in range(weight.shape[1]):
            if denominator[column_index] == 0:
                for row_index in range(weight.shape[0]):
                    weight[row_index][column_index] = 0
            else:
                for row_index in range(weight.shape[0]):
                    weight[row_index][column_index] = (
                        weight[row_index][column_index] / denominator[column_index]
                    )

        return tf.convert_to_tensor(weight)

    @classmethod
    def train_model(cls):
        """
        Trains a two-layer neural network to compress a given image.

        This method takes in the image data and uses it to train a two-layer neural
        network with a specified number of hidden units. The network is trained
        using gradient descent with a learning rate of 0.1. The network's weights are
        normalized by their Euclidean norms after each epoch. The training is
        stopped when the mean squared error between the predicted output and the
        target output is less than a specified maximum error.

        Args:
            None

        Returns:
            A tuple of two TensorFlow tensors, each representing the weights of
            the connections between the input and hidden layers, and between the
            hidden and output layers respectively.
        """
        error = cls.MAX_ERROR + 1
        epoch = 0

        layer1, layer2 = cls.initialize_layers()
        blocks = cls.generate_blocks()

        while error > cls.MAX_ERROR:
            error = 0
            epoch += 1
            for block in blocks:
                hidden_layer = tf.matmul(block, layer1)
                output_layer = tf.matmul(hidden_layer, layer2)
                diff = output_layer - block

                layer1 -= cls.LEARNING_RATE * tf.matmul(
                    tf.matmul(tf.transpose(block), diff), tf.transpose(layer2)
                )
                layer2 -= cls.LEARNING_RATE * tf.matmul(
                    tf.transpose(hidden_layer), diff
                )

                layer1 = cls.normalize(layer1)
                layer2 = cls.normalize(layer2)

            error = sum(
                tf.reduce_sum((block @ layer1 @ layer2 - block) ** 2)
                for block in blocks
            )
            print(f"Epoch {epoch} - Error: {error}")

        compression_ratio = (
            32 * cls.BLOCK_HEIGHT * cls.BLOCK_WIDTH * cls.total_blocks()
        ) / ((cls.INPUT_SIZE + cls.total_blocks()) * 32 * cls.HIDDEN_SIZE + 2)
        print(f"Compression Ratio: {compression_ratio}")
        return layer1, layer2

    @classmethod
    def mod_of_vector(cls, vector):
        return np.sqrt(np.sum(vector**2, axis=0))

    @classmethod
    def generate_blocks(cls):
        height, width = cls.get_image_dimensions()
        return cls.split_into_blocks(height, width).reshape(
            cls.total_blocks(), 1, cls.INPUT_SIZE
        )

    @classmethod
    def total_blocks(cls):
        height, width = cls.get_image_dimensions()
        return (height * width) // (cls.BLOCK_HEIGHT * cls.BLOCK_WIDTH)

    @classmethod
    def compress_image(cls, block_height, block_width):
        """
        Compresses an image by breaking it down into blocks of specified height
        and width, training a two-layer neural network to compress the blocks,
        and then applying the trained network to each block to obtain the
        compressed blocks.

        The compressed blocks are then rearranged into an image and displayed.

        Args:
            block_height (int): The height of each block of the image.
            block_width (int): The width of each block of the image.

        Returns:
            None
        """
        cls.BLOCK_HEIGHT, cls.BLOCK_WIDTH = block_height, block_width
        cls.INPUT_SIZE = cls.BLOCK_HEIGHT * cls.BLOCK_WIDTH * 3

        height, width = cls.get_image_dimensions()
        layer1, layer2 = cls.train_model()

        original_image = cls.load_image()
        compressed_blocks = [block @ layer1 @ layer2 for block in cls.generate_blocks()]
        compressed_image = np.clip(
            np.array(compressed_blocks).reshape(cls.total_blocks(), cls.INPUT_SIZE),
            -1,
            1,
        )

        cls.display_image(original_image)
        cls.display_image(cls.blocks_to_image_array(compressed_image, height, width))


if __name__ == "__main__":
    ImageCompressor.compress_image(8, 8)
