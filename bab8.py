import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from deap import base, creator, tools, algorithms

# Load and preprocess data
data = pd.read_csv('Rockpaper.csv')

# Selecting first 50 samples for classification
data = data.head(50)

# Assuming 'Score' is the target column and the rest are features
X = data.drop('Score', axis=1)
y = data['Score']

# Encode categorical variables if necessary
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Print some samples of the data for verification
print("Sample Data:")
print(X.head())
print(y.head())

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the fitness function
def evaluate(individual):
    weights = np.array(individual)
    weighted_sum = np.dot(X_train, weights)
    # Assuming binary classification, so we use a simple threshold
    predictions = (weighted_sum > np.median(weighted_sum)).astype(int)
    accuracy = (predictions == y_train).mean()
    return accuracy,

# Create the necessary classes for DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(X_train[0]))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Genetic Algorithm parameters
population = toolbox.population(n=50)
ngen = 40
cxpb = 0.5
mutpb = 0.2

# Print initial population fitness
print("Initial population fitness:")
for ind in population:
    fitness = evaluate(ind)
    ind.fitness.values = fitness
    print(fitness)

# Statistics to keep track of the evolution process
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Run the Genetic Algorithm
result, logbook = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=stats, halloffame=None, verbose=True)

# Print statistics for each generation
print("Logbook:")
for record in logbook:
    print(record)

# Evaluate on test set
best_individual = tools.selBest(result, 1)[0]
best_weights = np.array(best_individual)
weighted_sum_test = np.dot(X_test, best_weights)
test_predictions = (weighted_sum_test > np.median(weighted_sum_test)).astype(int)
test_accuracy = (test_predictions == y_test).mean()

print(f"Test Accuracy: {test_accuracy:.2f}")

# Plot the evolution of the fitness values
gen = logbook.select("gen")
fit_max = logbook.select("max")
fit_avg = logbook.select("avg")

plt.plot(gen, fit_max, label="Maximum Fitness")
plt.plot(gen, fit_avg, label="Average Fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(loc="lower right")
plt.title("Fitness Evolution")
plt.show()
