import pandas as pd
import k_nearest_neighbours as knn
import time

# Read the data
df = pd.read_csv("iris.data", header=None)

# Create the test data
test_data = [[5.6, 3.8, 1.4, 0.3],
             [8.2, 3.3, 6.0, 2.1],
             [5.7, 3.7, 1.5, 0.5],
             [5.5, 3.6, 1.3, 0, 4],
             [8.0, 3.4, 6.1, 2, 2],
             [8.1, 3.0, 6.0, 2.1],
             [7.8, 3.5, 5.8, 2.3],
             [6.2, 3.9, 6.2, 2.4],
             [6.4, 3.0, 4.8, 1.7],
             [4.4, 3.3, 5.5, 2.2]]

# Create a dataframe to store the results
test_results = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            columns=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150])

for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for n in [15, 30, 45, 60, 75, 90, 105, 120, 135, 150]:
        train_df = df.sample(n)
        classifier = knn.KNearestNeighbours(train_df.values.tolist(), k)
        predict_times = []
        for test in test_data:
            start_time = time.time()
            classifier.predict(test)
            predict_times.append(time.time() - start_time)
        test_results[n][k] = (sum(predict_times) / len(predict_times)) * 1000000

print(test_results)
