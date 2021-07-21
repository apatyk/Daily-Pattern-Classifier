# Daily Pattern Classifier
---
## Repository Structure

    common/                     // directory for common scripts and text files
        batch-unix.txt          // filenames for all recordings in CAD dataset
        meals-shimmer.txt       // information for all meals in CAD dataset
        loadfile.py             // Python file with functions for loading CAD dataset data
        testing.py              // Python file containing functions for evaluation
        training.py             // Python file with functions for training window-based classifier (Sharma 2020)

    GenerateSamples/            // code to generate daily samples for training daily pattern classifier
        GenerateSamples.ipynb   // Jupyter notebook for generating daily samples
        GenerateSamples.py      // Python program for generating daily samples
        LoadFiles.ipynb         // loads many daily sample text files and saves as combined .npy files
        SubmitGenSamplesJob.py  // script to run GenerateSamples on Palmetto cluster as a PBS job

    DailyPatternClassifier/     // code to train and evaluate the daily pattern classifier
        DailyPatternRNN.ipynb   // performs 5-fold cross validation for training AND testing (Jupyter notebook)
        TrainDailyPatternRNN.py // performs training for 5-fold cross validation
        TestDailyPatternRNN.py  // evaluates time and episode metrics post-hoc for 5-fold cross validation
        SubmitTrainRNNJob.py    // script to run TrainDailyPatternRNN on Palmetto cluster as a PBS job
        
## Code Description

1. 
