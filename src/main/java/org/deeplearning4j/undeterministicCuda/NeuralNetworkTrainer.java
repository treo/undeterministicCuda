package org.deeplearning4j.undeterministicCuda;

import org.deeplearning4j.nn.api.Model;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class NeuralNetworkTrainer {
    Logger D = LoggerFactory.getLogger(MaxEpochsTerminationCondition.class);


    public NeuralNetworkTrainer(DataSetIteratorsPack iterators, TerminationCondition... terminationConditions) {
        this.iterators = iterators;
        this.terminationConditions = terminationConditions;
        init();
    }

    protected DataSetIteratorsPack iterators;

    public DataSetIteratorsPack getIterators() {
        return iterators;
    }

    protected TerminationCondition[] terminationConditions;

    public TerminationCondition[] getTerminationConditions() {
        return terminationConditions;
    }

    private void init() {
        for (TerminationCondition terminationCondition : getTerminationConditions()) {
            terminationCondition.setTrainer(this);
        }
    }

    private void sanityCheck() {
        if (! getIterators().hasTrain()) {
            throw new RuntimeException("Train iterator is null!");
        }
        if (! getIterators().hasTest()) {
            throw new RuntimeException("Test iterator is null!");
        }
    }

    private boolean trainingRunning;

    public boolean isTrainingRunning() {
        return trainingRunning;
    }

    public void setTrainingRunning(boolean trainingRunning) {
        this.trainingRunning = trainingRunning;
    }

    public void run(Model model) {
        sanityCheck();
        long startTrainingTime = System.currentTimeMillis();
        setTrainingRunning(true);
        D.info("STARTING TRAINING");
        while (isTrainingRunning()) {
            D.info("Epoch number " + epochNumber);
            long startEpochTime = System.currentTimeMillis();
            fit(model);

            D.info("Calculating score...");
            if (calculatingTrainScore) {
                lastTrainScore = calculateTrainScore(model);
                D.info("Train score: " + lastTrainScore);
            }

            lastTestScore = calculateTestScore(model);
            D.info("Test score: " + lastTestScore);
//            D.info(new DataSetLossCalculator(getIterators().getTest(), true).calculateScore(model));

            // If score better then last best score save model as best model
            double testScoreDiff = lastBestTestScore - lastTestScore;
            if (testScoreDiff > 0) {
                saveBestModel(model);
                lastBestTrainScore = lastTrainScore;
                lastBestTestScore = lastTestScore;
            }
            D.info("Best score: " + lastBestTestScore);

            checkTerminalConditions();

            actualTrainingTime += System.currentTimeMillis() - startEpochTime;
            D.info("Epoch last: " + getTimeInReadableFormat(startEpochTime));
            D.info("\n");
            epochNumber++;
        }

        if (calculatingTrainScore) {
            D.info("Best train score: " + lastBestTrainScore);
        }
        D.info("Best test score: " + lastBestTestScore);

        if (getIterators().hasEval()) {
            D.info("Evaluating model...");
            double evalScore = calculateEvalScore(getBestModel());
            D.info("Eval score: " + evalScore);
            D.info("\n");
        }

        D.info("Training last for: " + getTimeInReadableFormat(startTrainingTime));
    }

    private void checkTerminalConditions() {
        for (TerminationCondition terminationCondition : getTerminationConditions()) {
            terminationCondition.checkConditions();
        }
    }

    public abstract Model getBestModel();

    protected abstract void fit(Model model);

    protected abstract double calculateTrainScore(Model model);

    protected abstract double calculateTestScore(Model model);

    protected abstract double calculateEvalScore(Model model);

    private String getTimeInReadableFormat(long startTime) {
        return String.valueOf((System.currentTimeMillis() - startTime) / 60000);
    }

    private int epochNumber;

    public int getEpochNumber() {
        return epochNumber;
    }

    private double lastBestTestScore = Double.MAX_VALUE;

    public double getLastBestTestScore() {
        return lastBestTestScore;
    }

    private double lastBestTrainScore = Double.MAX_VALUE;

    public double getLastBestTrainScore() {
        return lastBestTrainScore;
    }

    private double lastTrainScore;

    public double getLastTrainScore() {
        return lastTrainScore;
    }

    protected double lastTestScore;

    public double getLastTestScore() {
        return lastTestScore;
    }

    protected abstract void saveBestModel(Model model);

    private boolean calculatingTrainScore = true;

    public void setCalculatingTrainScore(boolean calculatingTrainScore) {
        this.calculatingTrainScore = calculatingTrainScore;
    }

    private long actualTrainingTime;

    public long getActualTrainingTime() {
        return actualTrainingTime;
    }
}