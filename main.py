import numpy as np
from preprocessing import load_process_data
from svm_ga import SVM_GA
from sklearn.metrics import accuracy_score


def rbf_kernel(X1, X2, gamma=0.1):
    sq_dists = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dists)

def main():
    X_train, X_test, y_train, y_test = load_process_data()

    X_train = X_train[:300]
    y_train = y_train[:300]

    C = 1.0
    gamma = 0.1
    pop_size = 40
    generations = 50

    print("Matricea Kernel")
    K = rbf_kernel(X_train, X_train, gamma)

    ga = SVM_GA(X_train, y_train, C=C, pop_size=pop_size)

    print(f"Incepem evolutia pentru {generations} generatii\n")

    pop = [ga.repair(np.random.uniform(0, C, len(y_train))) for _ in range(pop_size)]

    best_alpha = None
    best_fit = -np.inf

    for gen in range(generations):
        fits = [ga.fitness(ind, K) for ind in pop]

        current_best_idx = np.argmax(fits)
        if fits[current_best_idx] > best_fit:
            best_fit = fits[current_best_idx]
            best_alpha = pop[current_best_idx].copy()
            print(f"Generatia {gen}: Cel mai bun fitness = {best_fit:.4f}")

        new_pop = []
        for _ in range(pop_size):
            i1, i2 = np.random.choice(pop_size, 2, replace=False)
            p1 = pop[i1] if fits[i1] > fits[i2] else pop[i2]

            i3, i4 = np.random.choice(pop_size, 2, replace=False)
            p2 = pop[i3] if fits[i3] > fits[i4] else pop[i4]

            child = ga.crossover(p1, p2)
            child = ga.mutation(child, rate=0.05)
            new_pop.append(child)

        pop = new_pop

    sv_idx = np.where(best_alpha > 1e-5)[0]
    if len(sv_idx) > 0:
        b = np.mean(y_train[sv_idx] - np.dot(K[sv_idx], best_alpha * y_train))
    else:
        b = 0

    print("\nEvaluare pe setul de test\n")
    K_test = rbf_kernel(X_test, X_train, gamma)
    y_pred = np.sign(np.dot(K_test, best_alpha * y_train) + b)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acuratete finala: {acc * 100:.2f}%\n")


if __name__ == "__main__":
    main()