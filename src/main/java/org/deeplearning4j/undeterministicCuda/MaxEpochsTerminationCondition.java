package org.deeplearning4j.undeterministicCuda;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MaxEpochsTerminationCondition extends TerminationCondition {
    Logger D = LoggerFactory.getLogger(MaxEpochsTerminationCondition.class);

    public MaxEpochsTerminationCondition(int maxEpoch) {
        this.maxEpoch = maxEpoch;
    }

    private int maxEpoch;

    @Override
    protected String createTerminationMessage() {
        return "Stopping training. Max epochs " + maxEpoch + " reached.";
    }

    @Override
    public void checkConditions() {
        int epochNumber = getTrainer().getEpochNumber();
        if (epochNumber >= maxEpoch - 1) {
            terminateTraining();
        } else {
            if (epochNumber != 0) {
                long averageEpochTime = getTrainer().getActualTrainingTime() / epochNumber;
                long remainingTime = (maxEpoch - epochNumber - 1) * averageEpochTime;
                D.info("Approximate remaining time: " +remainingTime / 60000);
            }
        }
    }
}