# EventDetector
This is a project which implements a framework for detecting disaster events from Twitter.
This project has three parts:
- Part 1: The classification
- Part 2: Events detector
- Part 3: Interface module

# How to run:
Step 1:

We begin with the LSTM Folder. Go into LSTM, we run the lstm_T6.py. This module is going to grab datasets from dataset folder,  and create .hdf5 files and a tokenizer.pickle file. Choose the .hdf5 file with the best accuracy and tokenizer.pickle.

For Example: 

  lstm_T6_best_weights.02-0.9507.hdf5
  
  lstm_T6_best_weights.03-0.9407.hdf5
  
We will choose the first file, because its accuracy = 0.9507 compared with 0.9407 from the second file.

Step 2:

Paste .hdf5 and tokenizer.pickle into the classifier folder.
Open the demo folder and run the demo.py. When the terminal displays "THE INTERFACE IS CREATED, PLEASE RUN THE INTERFACE MODULE!!!",  please run the maps.py in the same folder.

Step 3: 

Open browser and access: 127.0.0.1:8050.
The interface module is going to display all disaster events from the U.S.A, and it will update automatically when the event detector catches up new events.

Thanks,
