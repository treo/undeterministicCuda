package org.deeplearning4j.undeterministicCuda;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class DataSetIteratorsPack {

    private DataSetIterator train, test, eval;

    public DataSetIteratorsPack(DataSetIterator train, DataSetIterator test, DataSetIterator eval) {
        this.train = train;
        this.test = test;
        this.eval = eval;
    }

    public DataSetIteratorsPack(DataSetIterator train, DataSetIterator test) {
        this(train, test, null);
    }

    public boolean hasTrain() {
        return getTrain() != null;
    }

    public boolean hasTest() {
        return getTest() != null;
    }

    public boolean hasEval() {
        return getEval() != null;
    }

    public DataSetIterator getTrain() {
        return train;
    }

    public DataSetIterator getTest() {
        return test;
    }

    public DataSetIterator getEval() {
        return eval;
    }
}
