package org.deeplearning4j.undeterministicCuda;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndexAll;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.io.IOException;

public class NeuralNetworkCreator {

    private static final int SEED = 123;
    private static final IUpdater UPDATER = new Nesterovs(0.01, 0.9);
    private static final double L2 = 1e-5;

    private static FineTuneConfiguration createFineTuneConf() {
        return FineTuneConfiguration.builder()
                .seed(SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(UPDATER)
                .l2(L2)
                .activation(Activation.IDENTITY)
                .weightInit(WeightInit.XAVIER)
                .convolutionMode(ConvolutionMode.Same)
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .build();
    }

    public static ComputationGraph tinyYoloPretrainedCG(int width, int height, int channels, INDArray priorBoxes) throws IOException {
        ZooModel<ComputationGraph> zooModel = TinyYOLO.builder().build();
        ComputationGraph pretrained = zooModel.initPretrained(PretrainedType.IMAGENET);

        ComputationGraph model = new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(createFineTuneConf())
                .removeVertexKeepConnections("conv2d_9")
                .removeVertexKeepConnections("outputs")
                .addLayer("conv2d_9",
                        new ConvolutionLayer.Builder(1,1)
                                .nIn(1024)
                                .nOut(priorBoxes.size(0) * (5 + 1))
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Same)
                                .build(),
                        "leaky_re_lu_8")
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .boundingBoxPriors(priorBoxes)
                                .build(),
                        "conv2d_9")
                .setOutputs("outputs")
                .setInputTypes(InputType.convolutional(height, width, channels))
                .build();
        model.getLayer("conv2d_9").setParam("W", pretrained.getLayer("conv2d_9").getParam("W").get(NDArrayIndex.interval(0, 30)));
        model.getLayer("conv2d_9").setParam("b", pretrained.getLayer("conv2d_9").getParam("b").get(new NDArrayIndexAll(), NDArrayIndex.interval(0, 30)));

        return model;
    }

    public static MultiLayerNetwork tinyYoloPretrainedMLN(int width, int height, int channels, INDArray priorBoxes) throws IOException {
        ComputationGraph graph = tinyYoloPretrainedCG(width, height, channels, priorBoxes);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(UPDATER)
                .l2(L2)
                .activation(Activation.IDENTITY)
                .weightInit(WeightInit.XAVIER)
                .convolutionMode(ConvolutionMode.Same)
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.USER_SPECIFIED)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .list()

                .layer(new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).nOut(16).hasBias(false).cudnnAlgoMode(ConvolutionLayer.AlgoMode.USER_SPECIFIED).cudnnBwdDataMode(ConvolutionLayer.BwdDataAlgo.ALGO_1).cudnnBwdFilterMode(ConvolutionLayer.BwdFilterAlgo.ALGO_1).build())
                .layer(new BatchNormalization.Builder().eps(0.001).useLogStd(false).build())
                .layer(new ActivationLayer.Builder(new ActivationLReLU(0.10000000149011612)).build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())

                .layer(new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).nOut(32).hasBias(false).cudnnAlgoMode(ConvolutionLayer.AlgoMode.USER_SPECIFIED).cudnnBwdDataMode(ConvolutionLayer.BwdDataAlgo.ALGO_1).cudnnBwdFilterMode(ConvolutionLayer.BwdFilterAlgo.ALGO_1).build())
                .layer(new BatchNormalization.Builder().eps(0.001).useLogStd(false).build())
                .layer(new ActivationLayer.Builder(new ActivationLReLU(0.10000000149011612)).build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())

                .layer(new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).nOut(64).hasBias(false).cudnnAlgoMode(ConvolutionLayer.AlgoMode.USER_SPECIFIED).cudnnBwdDataMode(ConvolutionLayer.BwdDataAlgo.ALGO_1).cudnnBwdFilterMode(ConvolutionLayer.BwdFilterAlgo.ALGO_1).build())
                .layer(new BatchNormalization.Builder().eps(0.001).useLogStd(false).build())
                .layer(new ActivationLayer.Builder(new ActivationLReLU(0.10000000149011612)).build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())

                .layer(new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).nOut(128).hasBias(false).cudnnAlgoMode(ConvolutionLayer.AlgoMode.USER_SPECIFIED).cudnnBwdDataMode(ConvolutionLayer.BwdDataAlgo.ALGO_1).cudnnBwdFilterMode(ConvolutionLayer.BwdFilterAlgo.ALGO_1).build())
                .layer(new BatchNormalization.Builder().eps(0.001).useLogStd(false).build())
                .layer(new ActivationLayer.Builder(new ActivationLReLU(0.10000000149011612)).build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())

                .layer(new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).nOut(256).hasBias(false).cudnnAlgoMode(ConvolutionLayer.AlgoMode.USER_SPECIFIED).cudnnBwdDataMode(ConvolutionLayer.BwdDataAlgo.ALGO_1).cudnnBwdFilterMode(ConvolutionLayer.BwdFilterAlgo.ALGO_1).build())
                .layer(new BatchNormalization.Builder().eps(0.001).useLogStd(false).build())
                .layer(new ActivationLayer.Builder(new ActivationLReLU(0.10000000149011612)).build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())

                .layer(new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).nOut(512).hasBias(false).cudnnAlgoMode(ConvolutionLayer.AlgoMode.USER_SPECIFIED).cudnnBwdDataMode(ConvolutionLayer.BwdDataAlgo.ALGO_1).cudnnBwdFilterMode(ConvolutionLayer.BwdFilterAlgo.ALGO_1).build())
                .layer(new BatchNormalization.Builder().eps(0.001).useLogStd(false).build())
                .layer(new ActivationLayer.Builder(new ActivationLReLU(0.10000000149011612)).build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(1, 1).build())

                .layer(new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).nOut(1024).hasBias(false).cudnnAlgoMode(ConvolutionLayer.AlgoMode.USER_SPECIFIED).cudnnBwdDataMode(ConvolutionLayer.BwdDataAlgo.ALGO_1).cudnnBwdFilterMode(ConvolutionLayer.BwdFilterAlgo.ALGO_1).build())
                .layer(new BatchNormalization.Builder().eps(0.001).useLogStd(false).build())
                .layer(new ActivationLayer.Builder(new ActivationLReLU(0.10000000149011612)).build())

                .layer(new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).nOut(1024).hasBias(false).cudnnAlgoMode(ConvolutionLayer.AlgoMode.USER_SPECIFIED).cudnnBwdDataMode(ConvolutionLayer.BwdDataAlgo.ALGO_1).cudnnBwdFilterMode(ConvolutionLayer.BwdFilterAlgo.ALGO_1).build())
                .layer(new BatchNormalization.Builder().eps(0.001).useLogStd(false).build())
                .layer(new ActivationLayer.Builder(new ActivationLReLU(0.10000000149011612)).build())

                .layer(new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).nOut(priorBoxes.size(0) * (5 + 1)).cudnnAlgoMode(ConvolutionLayer.AlgoMode.USER_SPECIFIED).cudnnBwdDataMode(ConvolutionLayer.BwdDataAlgo.ALGO_1).cudnnBwdFilterMode(ConvolutionLayer.BwdFilterAlgo.ALGO_1).build())
                .layer(new Yolo2OutputLayer.Builder().boundingBoxPriors(priorBoxes).build())

                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init(graph.params(), true);

        return model;
    }
}
