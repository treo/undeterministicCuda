package org.deeplearning4j.undeterministicCuda;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.ArrayList;

public class ProductDetectionTrainingTest {

    public static final int BATCH_SIZE = 8;
    public static final int GRID_W = 13;
    public static final int GRID_H = 13;
    public static final int GRID_PXS = 32;
    public static final int WIDTH = GRID_W * GRID_PXS;
    public static final int HEIGHT = GRID_H * GRID_PXS;
    public static final int MAX_EPOCHS_WITH_NO_IMPROVEMENT = 100;
    public static final int MAX_EPOCHS = 1000;

    public static final boolean DATA_AUGMENTATION = false;

    public static void main(String[] args) throws IOException {
        // create DataSetIterators
        DataSetIteratorsPack dataSetIteratorsPack = new DataSetIteratorsPack(createIteratorWithZeros(44), createIteratorWithZeros(15), createIteratorWithZeros(15));

        // create neural network
        INDArray priorBoxes = Nd4j.create(YOLO2.DEFAULT_PRIOR_BOXES);
        MultiLayerNetwork model = NeuralNetworkCreator.tinyYoloPretrainedMLN(WIDTH, HEIGHT, 3, priorBoxes);

        // train neural network
        MultiLayerNetworkTrainer trainer = new MultiLayerNetworkTrainer(dataSetIteratorsPack, new MaxEpochsTerminationCondition(MAX_EPOCHS));
        trainer.setCalculatingTrainScore(false);
        trainer.run(model);
        model = trainer.getBestModel();
    }

    private static ComputationGraph trainWithTheirTrainer(ComputationGraph model, DataSetIterator trainIterator, DataSetIterator testIterator) {
        EarlyStoppingConfiguration<ComputationGraph> conf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
                .epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(MAX_EPOCHS_WITH_NO_IMPROVEMENT, 0.0))
                .scoreCalculator(new DataSetLossCalculator(testIterator, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(new InMemoryModelSaver<>())
                .build();
        EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(conf, model, trainIterator);
        EarlyStoppingResult<ComputationGraph> result = trainer.fit();
        return result.getBestModel();
    }

    private static DataSetIterator createIteratorWithZeros(int n) {
        ArrayList<DataSet> result = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            result.add(new DataSet(Nd4j.zeros(1, 3, 416, 416), Nd4j.zeros(1, 5, 13, 13)));
        }
        return new ListDataSetIterator<>(result, BATCH_SIZE);
    }
}
