package org.deeplearning4j.undeterministicCuda;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class MultiLayerNetworkTrainer extends NeuralNetworkTrainer {

    public MultiLayerNetworkTrainer(DataSetIteratorsPack iterators, TerminationCondition... terminationConditions) {
        super(iterators, terminationConditions);
    }

    private MultiLayerNetwork bestModel;

    @Override
    public MultiLayerNetwork getBestModel() {
        return bestModel;
    }

    @Override
    protected void fit(Model model) {
        ((MultiLayerNetwork) model).fit(getIterators().getTrain());
    }

    @Override
    protected double calculateTrainScore(Model model) {
        return calculateScore((MultiLayerNetwork) model, getIterators().getTrain());
    }

    @Override
    protected double calculateTestScore(Model model) {
        return calculateScore((MultiLayerNetwork) model, getIterators().getTest());
    }

    @Override
    protected double calculateEvalScore(Model model) {
        return calculateScore((MultiLayerNetwork) model, getIterators().getEval());
    }

    private double calculateScore(MultiLayerNetwork graph, DataSetIterator dataSetIterator) {
        double score = 0;
        int n = 0;
        dataSetIterator.reset();
        while (dataSetIterator.hasNext()) {
            score += graph.score(dataSetIterator.next(), true);
            n++;
            System.gc();
        }
        return score / n;
    }

    @Override
    protected void saveBestModel(Model model) {
        bestModel = ((MultiLayerNetwork) model).clone();
    }
}
