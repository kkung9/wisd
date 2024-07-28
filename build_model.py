import pickle
import json
import os
import re
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

def prep_data():
    # get all files in the data folder
    files = os.listdir("data")

    # add contents of each file to a list
    list = []
    for file in files:
        if len(re.findall(".*\.jsonl.*", file)) == 1: # check format of file name
            data = json.load(open("data/" + file))
            assert data['units'] == {'length': 'foot', 'velocity': 'mph', 'acceleration': 'mph/s', 'angle': 'degree'}
            list.append(data)
        else:
            print("Warning: Could not read file: " + file)

    # make data frame from list
    df = pd.json_normalize(list)

    # rename and select relevant columns
    df = df.rename(columns = {
        "samples_ball": "ball_samples",
        "samples_bat": "bat_samples",
        "summary_acts.pitch.eventId": "pitch_id",
        "summary_acts.pitch.result": "pitch_result",
        "summary_acts.pitch.action": "pitch_action",
        "summary_acts.pitch.speed.mph": "pitch_speed",
        "summary_acts.pitch.spin.rpm": "pitch_spin",
        "summary_acts.hit.eventId": "hit_id",
        "summary_acts.hit.speed.mph": "hit_speed",
        "summary_acts.hit.spin.rpm": "hit_spin",
        "summary_score.runs.game.team1": "team1_runs",
        "summary_score.runs.game.team2": "team2_runs",
        "summary_score.runs.innings": "innings_runs",
        "summary_score.runs.play": "play_runs",
        "summary_score.outs.inning": "inning_outs",
        "summary_score.outs.play": "play_outs",
        "summary_score.count.balls.plateAppearance": "plate_appearance_balls",
        "summary_score.count.balls.play": "play_balls",
        "summary_score.count.strikes.plateAppearance": "plate_appearance_strikes",
        "summary_score.count.strikes.play": "play_strikes",
        "summary_acts.pitch.type": "pitch_type",
    })
    df = df[["events",
            "ball_samples",
            "bat_samples",
            "pitch_id",
            "pitch_result",
            "pitch_action",
            "pitch_speed",
            "pitch_spin",
            "hit_id",
            "hit_speed",
            "hit_spin",
            "team1_runs",
            "team2_runs",
            "innings_runs",
            "play_runs",
            "inning_outs",
            "play_outs",
            "plate_appearance_balls",
            "play_balls",
            "plate_appearance_strikes",
            "play_strikes",
            "pitch_type"
            ]]

    # mask cells that effectively have no data
    df.events = df.events.mask(df.events.str.len() == 0)
    df.bat_samples = df.bat_samples.mask(df.bat_samples.str.len() <= 1)

    # extract nested data from ball samples column into its own data frame
    # balls = df[["pitch_id", "hit_id", "ball_samples", "pitch_result", "pitch_speed", "pitch_spin"]]
    def extract_ball_data(data_row):
        balls = data_row.ball_samples
        expanded = pd.json_normalize(balls)
        expanded[["x_pos", "y_pos", "z_pos"]] = pd.DataFrame(expanded.pos.tolist())
        # fastest way I could think of dealing with the NA rows, ideally would probably just fit a quadratic to figure out vel and acc
        vels = np.full((len(expanded), 3), pd.NA)
        vels[pd.notna(expanded.vel)] = expanded[pd.notna(expanded.vel)].vel.tolist()
        expanded[["x_vel", "y_vel", "z_vel"]] = vels
        accs = np.full((len(expanded), 3), pd.NA)
        accs[pd.notna(expanded.acc)] = expanded[pd.notna(expanded.acc)].acc.tolist()
        expanded[["x_acc", "y_acc", "z_acc"]] = accs
        expanded[["hit_id"]] = data_row.hit_id
        expanded[["pitch_id"]] = data_row.pitch_id
        return expanded
    balls = pd.DataFrame()
    for i in range(len(df)):
        balls = pd.concat([balls, extract_ball_data(df.iloc[i])])

    has_balls = balls[pd.notna(balls.pitch_id)].reset_index()
    relevant = df[df.pitch_result != "Pickoff"]
    initial = has_balls.loc[has_balls.groupby([has_balls.pitch_id]).time.idxmin()]
    relevant = pd.merge(relevant, initial, on = "pitch_id")[["pitch_id", "pitch_speed", "pitch_spin", "x_pos", "y_pos", "z_pos", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "pitch_result"]]
    relevant = relevant.drop(np.where(pd.isna(relevant))[0])
    past_plate = initial.copy()

    # extrapolate where x_pos and z_pos coordinates would be when y_pos = 0
    over = pd.DataFrame()
    # test = []
    for id in np.unique(balls.pitch_id[pd.notna(balls.pitch_id)]):
        pitch = balls[balls.pitch_id == id].reset_index()
        try:
            almost_there = np.where(abs(np.diff(pitch.y_acc[pd.notna(pitch.y_acc)])) > 1)[0][0]
        except:
            almost_there = len(pitch)
        try:
            y_model = np.polyfit(pitch.time.loc[:almost_there].values, pitch.y_pos.loc[:almost_there].values, 1)
            t = -y_model[1] / y_model[0]
            x_model = np.polyfit(pitch.time.loc[:almost_there].values, pitch.x_pos.loc[:almost_there].values, 2)
            x_pos = x_model[0] * t * t + x_model[1] * t + x_model[2]
            z_model = np.polyfit(pitch.time.loc[:almost_there].values, pitch.z_pos.loc[:almost_there].values, 2)
            z_pos = z_model[0] * t * t + z_model[1] * t + z_model[2]
        except:
            continue

        row = pd.DataFrame({"pitch_id": [id], "z0_extrap": [z_pos], "x0_extrap": [x_pos], "time0_extrap": [t]})
        over = pd.concat([row, over], ignore_index=True)
        # test.append(almost_there)

    past_plate = pd.merge(past_plate, over, on = "pitch_id")[["pitch_id", "x_pos", "y_pos", "z_pos", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "x0_extrap", "z0_extrap", "time0_extrap"]]
    past_plate = pd.merge(past_plate, df, on = "pitch_id")[["pitch_id", "pitch_speed", "pitch_spin", "pitch_result", "x_pos", "y_pos", "z_pos", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "x0_extrap", "z0_extrap", "time0_extrap"]]

    past_plate = past_plate[past_plate.pitch_result != "Pickoff"]
    past_plate = past_plate.drop(np.where(pd.isna(past_plate))[0])
    past_plate = past_plate[past_plate.pitch_id != "09613812-5231-471b-860e-ab8b5a30b40d"].reset_index()

    return df, balls, past_plate

df, balls, data = prep_data()

init = data[["pitch_speed", "pitch_spin", "x_pos", "y_pos", "z_pos", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc"]]
x0 = data[["x0_extrap"]]
z0 = data[["z0_extrap"]]
result = data[["pitch_result"]]

model1_x = LinearRegression()
model1_x.fit(init, x0)
model1_z = LinearRegression()
model1_z.fit(init, z0)

model2 = KNeighborsClassifier(n_neighbors = 35)
model2.fit(data[["x0_extrap", "z0_extrap"]], result.values.ravel())

with open('models/model1_x.pkl','wb') as f:
    pickle.dump(model1_x,f)
with open('models/model1_z.pkl','wb') as f:
    pickle.dump(model1_z,f)
with open('models/model2.pkl','wb') as f:
    pickle.dump(model2,f)

