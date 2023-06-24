import tensorflow as tf


def main():
    tensor = tf.sparse.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]
    )
    print(tensor)


if __name__ == "__main__":
    main()
