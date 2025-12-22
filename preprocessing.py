import pandas
import numpy
from sklearn.model_selection import train_test_split

def load_process_data():
    columns = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat']

    df = pandas.read_csv('data/mushrooms.csv', header=None, names=columns)

    y_raw = df['class']
    x_raw = df.drop('class', axis=1)

    y = numpy.where(y_raw == 'p', 1, -1)
    x = pandas.get_dummies(x_raw, dtype=int)
    x = x.values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print(f"{x_train.shape[0]} train samples\n")
    print(f"{x_test.shape[0]} test samples\n")
    print(f"{x_train.shape[1]} features\n")
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    load_process_data()

