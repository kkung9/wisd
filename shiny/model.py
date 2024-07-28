import pickle
import pandas as pd

def predict(dict):
    inputs = pd.DataFrame(dict, index = [0])
    with open('models/model1_x.pkl', 'rb') as f:
        model1_x = pickle.load(f)
    with open('models/model1_z.pkl', 'rb') as f:
        model1_z = pickle.load(f)
    with open('models/model2.pkl', 'rb') as f:
        model2 = pickle.load(f)
    pred_x = model1_x.predict(inputs)
    pred_z = model1_z.predict(inputs)
    pred_coords = pd.concat([pd.DataFrame(pred_x, columns = ["x0_extrap"]), pd.DataFrame(pred_z, columns = ["z0_extrap"])], axis = 1)
    pred = model2.predict(pred_coords)
    return pred

def get_bounds(col):
    bounds = {
        "pitch_speed": [70, 98, 87],
        "pitch_spin": [10, 3350, 2270],
        "x_pos": [-4, 3, -2],
        "y_pos": [41, 56, 51],
        "z_pos": [3, 7, 5],
        "x_vel": [-8, 11, 2],
        "y_vel": [-97, -69, -87],
        "z_vel": [-9, 6, -2],
        "x_acc": [-16, 15, -3],
        "y_acc": [10, 26, 18],
        "z_acc": [-30, -5, -17]
    }
    return bounds[col]