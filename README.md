## Pitch Outcome Prediction

### Running the Program
__Dependencies__:
* shiny
* numpy
* matplotlib
* pandas
* sklearn

To run the program locally, run the _app.py_ file in the _shiny_ folder. \
From the root directory: \
`shiny run --reload --launch-browser shiny/app.py`

### Behind the Program
The _build_model.py_ and _make_images.py_ files create the _models_ and _images_ folders, respectively. These folders are relied upon by the program, but these files do not need to be re-run. They are included to provide further detail on how the model and content in the program was created. \
Both of these files require the user to download all data from the dropbox into a folder called _data_.
