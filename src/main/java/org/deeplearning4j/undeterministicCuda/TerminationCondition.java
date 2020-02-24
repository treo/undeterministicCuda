package org.deeplearning4j.undeterministicCuda;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class TerminationCondition {
    Logger D = LoggerFactory.getLogger(MaxEpochsTerminationCondition.class);


    protected NeuralNetworkTrainer trainer;

    protected NeuralNetworkTrainer getTrainer() {
        return trainer;
    }

    public void setTrainer(NeuralNetworkTrainer trainer) {
        this.trainer = trainer;
    }

    protected abstract String createTerminationMessage();

    public void terminateTraining() {
        getTrainer().setTrainingRunning(false);
        D.info(createTerminationMessage());
    }

    public abstract void checkConditions();
}