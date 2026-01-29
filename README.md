# lstm-river-stage-forecasting

This is example code for training, running, and evaluating basin-specific LSTM models to predict future river stage height. Input data can be found here: https://zenodo.org/records/18406565.

## Project Structure

```
geospatial_app/
├── EvalCode/
│   ├── stats_driver.py      # Example code for calculating avg stats across multiple forecast times and forecast types (input data needed)
│   └── stats_functions.py   # Helper functions used by stats_driver.py
│
├── RunCode/
│   ├── hFunc.py             # Helper functions for run_LSTM.py
│   ├── read_gauge.py        # Code to read gauge data
│   ├── river_info.json      # Dictionary containing river/gauge info
│   └── run_LSTM.py          # Code to run a LSTM forecast (input data needed)
│
├── TrainingCode/             
│   ├── BuildModel.py        # Code to build the LSTM model
│   ├── read_gauge.py        # Code to read gauge data
│   └── run_LSTM.py          # Code to train LSTM models (input via Zenodo link above)

```
