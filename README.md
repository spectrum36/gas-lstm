a personal project of mine, this has all of the files for it, I'm collecting data and using it to train an lstm to predict prices

what each file does:
1. adfuller.py, I used this to calculate stationarity on gas prices
2. crude.py, grabs data on petroleum products from eia.gov and puts it in influxdb for future use
3. priceDf.py, reads from the influxdb database to produce a dataframe, saved as a csv, with all of the data
4. weatherData.py, grabs weather data from all state capitals, stores the dataframs as a csv
5. locations.py, supports weatherData.py, it grabs state capital, state pairs off of wikipedia for use with geopy
6. sequencePercent.py, takes the two csvs and returns training and eval data sequences, used for the training files
7. torch1.py, trains one model and produces a graph, will be removed soon once I feel comfortable with my new system
8. torchTest.py, iterates over a set of hyperparameters, will also be removed soon
9. torchSearch.py, iterates over a set of hyperparamers
10. torchSingle.py, trains one model and produces a graph
11. modelRun.py, holds the code for training for the previous two files
