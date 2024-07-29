import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from build_model import prep_data

df, balls, past_plate = prep_data()

def make_y0_extrap_plot():
    idx = np.random.randint(0, len(past_plate), size = 10)
    for i in range(len(idx)):
        pitch_id = past_plate.iloc[idx[i]].pitch_id
        recorded = balls[balls.pitch_id == pitch_id]
        extrapolated = past_plate[past_plate.pitch_id == pitch_id]

        fig, axs = plt.subplots(2, 1)
        r_x = axs[0].scatter(recorded.time, recorded.x_pos, label = "Recorded")
        e_x = axs[0].scatter(extrapolated.time0_extrap, extrapolated.x0_extrap, label = "Extrapolated")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("X Position")
        axs[0].legend(handles = [r_x, e_x])
        r_z = axs[1].scatter(recorded.time, recorded.z_pos, label = "Recorded")
        e_z = axs[1].scatter(extrapolated.time0_extrap, extrapolated.z0_extrap, label = "Extrapolated")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Z Position")
        axs[1].legend(handles = [r_z, e_z])
        plt.suptitle("Pitch " + pitch_id)
        plt.savefig("images/extrapolated_pos_" + str(i))

def model1_performance():
    scores_x = []
    scores_z = []
    for i in range(20):
        train, test = train_test_split(past_plate)
        X_train = train.drop(["x0_extrap", "z0_extrap", "time0_extrap", "pitch_id", "pitch_result"], axis = 1)
        y_train_x = train["x0_extrap"]
        y_train_z = train["z0_extrap"]
        X_test = test.drop(["x0_extrap", "z0_extrap", "time0_extrap", "pitch_id", "pitch_result"], axis = 1)
        y_test_x = test["x0_extrap"]
        y_test_z = test["z0_extrap"]

        model_x = LinearRegression()
        model_x.fit(X_train, y_train_x)
        scores_x.append(model_x.score(X_test, y_test_x))

        model_z = LinearRegression()
        model_z.fit(X_train, y_train_z)
        scores_z.append(model_z.score(X_test, y_test_z))

    print("X model mean R squared: " + str(np.mean(scores_x)))
    print("X model standard deviation R squared: " + str(np.std(scores_x)))
    print("Z model mean R squared: " + str(np.mean(scores_z)))
    print("Z model standard deviation R squared: " + str(np.std(scores_z)))

def make_results_plot():
    fig = plt.figure()
    b = plt.scatter(past_plate[past_plate.pitch_result == "Ball"].x0_extrap, past_plate[past_plate.pitch_result == "Ball"].z0_extrap, c = "orange", alpha = 0.5, label = "Ball")
    s = plt.scatter(past_plate[past_plate.pitch_result == "Strike"].x0_extrap, past_plate[past_plate.pitch_result == "Strike"].z0_extrap, c = "blue", alpha = 0.5, label = "Strike")
    h = plt.scatter(past_plate[past_plate.pitch_result == "HitIntoPlay"].x0_extrap, past_plate[past_plate.pitch_result == "HitIntoPlay"].z0_extrap, c = "red", alpha = 0.5, label = "Hit")
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")
    plt.ylabel("Outcome Based on Position Over Front of Plate")
    plt.legend(handles = [b, h, s])
    plt.savefig("images/color_results")

def model_comparison():
    scores_dt = []
    scores_gbm = []
    scores_knn = []
    for i in range(10):
        train, test = train_test_split(past_plate[["x0_extrap", "z0_extrap", "pitch_result"]])
        X_train = train[["x0_extrap", "z0_extrap"]]
        y_train = train["pitch_result"]
        X_test = test[["x0_extrap", "z0_extrap"]]
        y_test = test["pitch_result"]

        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        scores_dt.append(dt.score(X_test, y_test))

        parameters = {'learning_rate': np.arange(0.48, 0.64, 0.01)}
        gbm = GridSearchCV(GradientBoostingClassifier(), parameters)
        gbm.fit(X_train, y_train)
        scores_gbm.append(gbm.score(X_test, y_test))

        parameters = {"n_neighbors": np.arange(10, 50, 2)}
        knn = GridSearchCV(KNeighborsClassifier(), parameters)
        knn.fit(X_train, y_train)
        scores_knn.append(knn.score(X_test, y_test))
    
    plt.boxplot([scores_dt, scores_gbm, scores_knn])
    plt.xticks([1,2,3], ["decision tree", "gradient boosting machine", "k nearest neighbors"])
    plt.ylabel("R squared")
    plt.title("Model Performance Over 10 Trials")
    plt.savefig("images/compare_models")

def pick_n_neighbors():
    n = []
    for i in range(50):
        train, test = train_test_split(past_plate[["x0_extrap", "z0_extrap", "pitch_result"]])
        X = train[["x0_extrap", "z0_extrap"]]
        y = train["pitch_result"]
        parameters = {"n_neighbors": np.arange(10, 50, 1)}
        knn = GridSearchCV(KNeighborsClassifier(), parameters)
        knn.fit(X, y)
        n.append(knn.best_params_["n_neighbors"])
    
    plt.hist(n)
    plt.xlabel("Number of Neighbors")
    plt.xlabel("Frequency")
    plt.title("Ideal Number of Neighbors Over 50 Trials")
    plt.savefig("images/knn")

    print("Mean best number of neighbors: " + str(np.mean(n)))

def final_performance():
    scores = []
    for i in range(20):
        print(i)
        train, test = train_test_split(past_plate)
        X1_train = train[["pitch_speed", "pitch_spin", "x_pos", "y_pos", "z_pos", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc"]]
        X2_train_x = train[["x0_extrap"]]
        X2_train_z = train[["z0_extrap"]]
        y_train = train["pitch_result"]
        X1_test = test[["pitch_speed", "pitch_spin", "x_pos", "y_pos", "z_pos", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc"]]
        y_test = test["pitch_result"]

        model1_x = LinearRegression()
        model1_x.fit(X1_train, X2_train_x.squeeze())
        model1_z = LinearRegression()
        model1_z.fit(X1_train, X2_train_z.squeeze())

        model2 = KNeighborsClassifier(n_neighbors = 35)
        model2.fit(pd.concat([X2_train_x.x0_extrap, X2_train_z.z0_extrap], axis = 1), y_train)

        X2_pred_x = model1_x.predict(X1_test)
        X2_pred_z = model1_z.predict(X1_test)

        scores.append(model2.score(pd.concat([pd.DataFrame(X2_pred_x, columns = ["x0_extrap"]), pd.DataFrame(X2_pred_z, columns = ["z0_extrap"])], axis = 1), y_test))
    
    print("Mean R squared: " + str(np.mean(scores)))
    print("Standard deviation R squared: " + str(np.std(scores)))

# make_y0_extrap_plot()
# make_results_plot()
# model1_performance()
# model_comparison()
# pick_n_neighbors()
# final_performance()