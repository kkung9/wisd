from shiny import App, render, ui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from model import predict, get_bounds

var_names = {
                "x_acc": "Initial X Acceleration",
                "y_acc": "Initial Y Acceleration",
                "z_acc": "Initial Z Acceleration",
                "x_vel": "Initial X Velocity",
                "y_vel": "Initial Y Velocity",
                "z_vel": "Initial Z Velocity",
                "x_pos": "Initial X Position",
                "y_pos": "Initial Y Position",
                "z_pos": "Initial Z Position",
                "pitch_speed": "Initial Speed",
                "pitch_spin": "Initial Spin",
            },

app_ui = ui.page_fluid(
    ui.navset_tab(
        ui.nav_panel("Prediction",
            ui.panel_title("Pitch Outcome Prediction"),
            ui.row(
                ui.input_slider("speed", "Initial Speed", get_bounds("pitch_speed")[0], get_bounds("pitch_speed")[1], get_bounds("pitch_speed")[2]),
                ui.input_slider("spin", "Initial Spin", get_bounds("pitch_spin")[0], get_bounds("pitch_spin")[1], get_bounds("pitch_spin")[2]),
            ),
            ui.row(
                ui.input_slider("xpos", "Initial X Position", get_bounds("x_pos")[0], get_bounds("x_pos")[1], get_bounds("x_pos")[2]),
                ui.input_slider("ypos", "Initial Y Position", get_bounds("y_pos")[0], get_bounds("y_pos")[1], get_bounds("y_pos")[2]),
                ui.input_slider("zpos", "Initial Z Position", get_bounds("z_pos")[0], get_bounds("z_pos")[1], get_bounds("z_pos")[2]),
            ),
            ui.row(
                ui.input_slider("xvel", "Initial X Velocity", get_bounds("x_vel")[0], get_bounds("x_vel")[1], get_bounds("x_vel")[2]),
                ui.input_slider("yvel", "Initial Y Velocity", get_bounds("y_vel")[0], get_bounds("y_vel")[1], get_bounds("y_vel")[2]),
                ui.input_slider("zvel", "Initial Z Velocity", get_bounds("z_vel")[0], get_bounds("z_vel")[1], get_bounds("z_vel")[2]),
            ),
            ui.row(
                ui.input_slider("xacc", "Initial X Acceleration", get_bounds("x_acc")[0], get_bounds("x_acc")[1], get_bounds("x_acc")[2]),
                ui.input_slider("yacc", "Initial Y Acceleration", get_bounds("y_acc")[0], get_bounds("y_acc")[1], get_bounds("y_acc")[2]),
                ui.input_slider("zacc", "Initial Z Acceleration", get_bounds("z_acc")[0], get_bounds("z_acc")[1], get_bounds("z_acc")[2]),
            ),
            ui.output_text_verbatim("txt"),
            ui.row(
                ui.input_select("variable1", "Choose First Feature",
                    {
                        "pitch_speed": "Initial Speed",
                        "pitch_spin": "Initial Spin",
                        "x_pos": "Initial X Position",
                        "y_pos": "Initial Y Position",
                        "z_pos": "Initial Z Position",
                        "x_vel": "Initial X Velocity",
                        "y_vel": "Initial Y Velocity",
                        "z_vel": "Initial Z Velocity",
                        "x_acc": "Initial X Acceleration",
                        "y_acc": "Initial Y Acceleration",
                        "z_acc": "Initial Z Acceleration",
                    },
                ),
                ui.input_select("variable2", "Choose Second Feature",
                    {
                        "x_acc": "Initial X Acceleration",
                        "y_acc": "Initial Y Acceleration",
                        "z_acc": "Initial Z Acceleration",
                        "x_vel": "Initial X Velocity",
                        "y_vel": "Initial Y Velocity",
                        "z_vel": "Initial Z Velocity",
                        "x_pos": "Initial X Position",
                        "y_pos": "Initial Y Position",
                        "z_pos": "Initial Z Position",
                        "pitch_speed": "Initial Speed",
                        "pitch_spin": "Initial Spin",
                    },
                ),
            ),
            ui.output_plot("plot"),
        ),
        ui.nav_panel("Model",
            ui.panel_title("Creation of the Model"),
            ui.p("A pitcher can control or inflluence features of the pitch such as initial speed, spin, position, velocity, and acceleration. The model uses these features to predict the position of the ball when it reaches the front of the plate. This model then aims to predict the tendencies of the outcomes of the pitch (ball, hit, or strike) based on those coordinates."),
            ui.p("First, the position of the ball over the front of the plate was determined. However, not every pitch reaches the front of the plate before making contact with the bat. Additionally, coordinate data is not necessarily recorded at the exact moment that the y coordinate was 0. Therefore, the x and z coordinates when the y coordinate was zero was extraploted. A random handful of pitches' recorded positions and the extrapolated coordinate when the y coordinate is zero are shown below."),
            ui.output_image("y0_extrap"),
            ui.p("While the extrapolated coordinates are not technically data, based on the trajectories of the balls, it is reasonable to assume that this is where the ball would have ended up if not acted upon by the bat. These coordinates can be effectively predicted using the initial features of the pitch. Over 20 trials with 75% of the data used as training data and 25% of the data used as test data, the average R-squared value for the x coordinate was 0.984 with a standard deviation of 0.0016, and the average R-squared value for the z coordinate was 0.983 with a standard deviation of 0.0119."),
            ui.p("The position over the plate was then used to predict the outcome of the pitch. These variables are less cleanly related, as there is more unpredictability that is out of the pitcher's control."),
            ui.output_image("color_results"),
            ui.p("Three types of models were attempted to predict the pitch result based on the coordinates: Decision Tree, Gradient Boosting Machine, K Nearest Neighbors. Over 5 trials using 75% of the data as training data and 25% of the data as test data, K Nearest Neighbors tended to perform the best."),
            ui.output_image("model2_compare"),
            ui.p("After deciding to go with the K Nearest Neighbors Model, we want to further optimize how many neighbors to use. For this, 50 trials were run using a Grid Search. While there was a lot of variation, th mean best number of neighbors was 34.96, so we will use 35 neighbors to fit the model."),
            ui.output_image("num_neighbors"),
            ui.p("To make the final model, a KNN model was fit using 35 neighbors."),
        )
    )
)

def server(input, output, session):
    @render.text
    def txt():
        pred = predict({
            "pitch_speed": input.speed(),
            "pitch_spin": input.spin(),
            "x_pos": input.xpos(),
            "y_pos": input.ypos(),
            "z_pos": input.zpos(),
            "x_vel": input.xvel(),
            "y_vel": input.yvel(),
            "z_vel": input.zvel(),
            "x_acc": input.xacc(),
            "y_acc": input.yacc(),
            "z_acc": input.zacc()
        })[0]
        return f"The predicted outcome is {pred}."

    @render.plot
    def plot():
        if input.variable1() == input.variable2():
            fig, ax = plt.subplots()
            ax.text(-0.175, 0, "Please set two different features.")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            return fig
        dict = {
            "pitch_speed": input.speed(),
            "pitch_spin": input.spin(),
            "x_pos": input.xpos(),
            "y_pos": input.ypos(),
            "z_pos": input.zpos(),
            "x_vel": input.xvel(),
            "y_vel": input.yvel(),
            "z_vel": input.zvel(),
            "x_acc": input.xacc(),
            "y_acc": input.yacc(),
            "z_acc": input.zacc()
        }
        x1_ball = []
        x2_ball = []
        x1_hit = []
        x2_hit = []
        x1_strike = []
        x2_strike = []
        for i in np.linspace(get_bounds(input.variable1())[0], get_bounds(input.variable1())[1], num = 10):
            for j in np.linspace(get_bounds(input.variable2())[0], get_bounds(input.variable2())[1], num = 10):
                dict[input.variable1()] = i
                dict[input.variable2()] = j
                pred = predict(dict)[0]
                if pred == "Ball":
                    x1_ball.append(i)
                    x2_ball.append(j)
                elif pred == "HitIntoPlay":
                    x1_hit.append(i)
                    x2_hit.append(j)
                elif pred == "Strike":
                    x1_strike.append(i)
                    x2_strike.append(j)

        inputs = [input.variable1(), input.variable2()]
        labels = ["", ""]
        for i in [0, 1]:
            if inputs[i] == "pitch_speed":
                labels[i] = "Initial Speed"
            elif inputs[i] == "pitch_spin":
                labels[i] = "Initial Spin"
            elif inputs[i] == "x_pos":
                labels[i] = "Initial X Position"
            elif inputs[i] == "y_pos":
                labels[i] = "Initial Y Position"
            elif inputs[i] == "z_pos":
                labels[i] = "Initial Z Position"
            elif inputs[i] == "x_vel":
                labels[i] = "Initial X Velocity"
            elif inputs[i] == "y_vel":
                labels[i] = "Initial Y Velocity"
            elif inputs[i] == "z_vel":
                labels[i] = "Initial Z Velocity"
            elif inputs[i] == "x_acc":
                labels[i] = "Initial X Acceleration"
            elif inputs[i] == "y_acc":
                labels[i] = "Initial Y Acceleration"
            elif inputs[i] == "z_acc":
                labels[i] = "Initial Z Acceleration"

        fig, ax = plt.subplots()
        b = ax.scatter(x1_ball, x2_ball, c = "orange", label = "Ball")
        h = ax.scatter(x1_hit, x2_hit, c = "red", label = "Hit")
        s = ax.scatter(x1_strike, x2_strike, c = "blue", label = "Strike")
        ax.legend(handles = [b, h, s])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title("Predicted Pitch Outcome (if all other features stay constant)")
        return fig
    
    @render.plot
    def y0_extrap():
        nums = np.random.randint(0, 10, size = 3)
        fig, ax = plt.subplots(1, 3)
        for i in range(3):
            im = img.imread("images/extrapolated_pos_" + str(nums[i]) + ".png")
            ax[i].imshow(im)
            ax[i].set_axis_off()
        return fig
    
    @render.plot
    def color_results():
        fig, ax = plt.subplots()
        im = img.imread("images/color_results.png")
        ax.imshow(im)
        ax.set_axis_off()

    @render.plot
    def model2_compare():
        fig, ax = plt.subplots()
        im = img.imread("images/compare_models.png")
        ax.imshow(im)
        ax.set_axis_off()

    @render.plot
    def num_neighbors():
        fig, ax = plt.subplots()
        im = img.imread("images/knn.png")
        ax.imshow(im)
        ax.set_axis_off()

app = App(app_ui, server)
