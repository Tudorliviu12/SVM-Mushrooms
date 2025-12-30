import numpy as np
import random


class SVM_GA:
    def __init__(self, X, y, C=1.0, pop_size=50):
        self.X = X
        self.y = y
        self.C = C
        self.pop_size = pop_size
        self.n_samples = X.shape[0]

    def fitness(self, alpha, K):
        term1 = np.sum(alpha)
        ay = alpha * self.y
        term2 = 0.5 * np.dot(ay.T, np.dot(K, ay))
        return term1 - term2

    def repair(self, alpha):
        for _ in range(10):
            alpha = np.clip(alpha, 0, self.C)
            current_sum = np.sum(alpha * self.y)
            if abs(current_sum) < 0.001:
                break
            idx = random.randint(0, self.n_samples - 1)
            correction = current_sum/self.y[idx]
            alpha[idx] = alpha[idx] - correction

        return alpha

    def crossover(self, p1, p2):
        gamma = random.random()
        return gamma * p1 + (1 - gamma) * p2

    def mutation(self, alpha, rate=0.01):
        for i in range(len(alpha)):
            if random.random() < rate:
                alpha[i] = random.uniform(0, self.C)
        return self.repair(alpha)