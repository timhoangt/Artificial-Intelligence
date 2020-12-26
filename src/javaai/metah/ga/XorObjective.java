/*
 Copyright (c) Ron Coleman

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/*
Timothy Hoang 11/3/20
 */
package javaai.metah.ga;

import org.encog.ml.CalculateScore;
import org.encog.ml.MLMethod;
import org.encog.ml.genetic.genome.DoubleArrayGenome;
import org.encog.ml.genetic.genome.IntegerArrayGenome;

import java.util.Random;

import static javaai.util.Helper.asDouble;
import static javaai.util.Helper.asInt;

/**
 * This class calculates the fitness of an individual chromosome or phenotype.
 */
class XorObjective implements CalculateScore {

    /**
     * Calculates the fitness.
     * @param phenotype Individual
     * @return Objective
     */
    @Override
    public double calculateScore(MLMethod phenotype) {
        DoubleArrayGenome genome = (DoubleArrayGenome) phenotype;

        /*int x = asInt(genome);

        double y = f(x);

        return y;*/

        double[] x = genome.getData();

        double y = getFitness(x);

        return y;
    }

    /**
     * Specifies the objective
     * @return True to minimize, false to maximize.
     */
    @Override
    public boolean shouldMinimize() {
        return true;
    }

    /**
     * Specifies the threading approach.
     * @return True to use single thread, false for multiple threads
     */
    @Override
    public boolean requireSingleThreaded() {
        return true;
    }

    /**
     * Objective function
     * @param x Domain parameter.
     * @return y
     */
    protected int f(int x) {
        return (x - 3)*(x - 3);
    }

    public final static boolean DEBUGGING = false;
    public final static String TEAM = "Timothy Hoang";
    public final static int NUM_WEIGHTS = 10;
    public final static double RANGE_MAX = 10.0;
    public final static double RANGE_MIN = -10.0;
    protected static Random ran = null;
    static {
        long seed = System.nanoTime();
        if(DEBUGGING)
            seed = TEAM.hashCode();
        ran = new Random(seed);
    }

    public final static double[][] XOR_INPUTS = {
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 1.0},
            {0.0, 1.0, 0.0},
            {0.0, 1.0, 1.0},
            {1.0, 0.0, 0.0},
            {1.0, 0.0, 1.0},
            {1.0, 1.0, 0.0},
            {1.0, 1.0, 1.0}
    };
    public final static double[][] XOR_IDEALS = {
            {1.0},
            {1.0},
            {0.0},
            {1.0},
            {0.0},
            {0.0},
            {0.0},
            {1.0}};

    /**
     * Gets a random weight.
     * @return double
     */
    public static double getRandomWeight() {
        double wt = ran.nextDouble() * (RANGE_MAX - RANGE_MIN) + RANGE_MIN;
        return wt;
    }

    /**
     * Calculates the sigmoid function activation function.
     * @param z double
     * @return double
     */
    protected double sigmoid(double z){
        double sigz =  1.0 / (1 + Math.exp (-1 * (z)));
        return sigz;
    }

    /**
     * Calculates the ANN output.
     * @param x1 double the XOR input
     * @param x2 double the XOR input
     * @param ws double[] the interneuron weights
     * @return double
     */
    public double feedforward(double x1, double x2, double x3, double[] ws){
        double w1 = ws[0];
        double w2 = ws[1];
        double w3 = ws[2];
        double w4 = ws[3];
        double w5 = ws[4];
        double w6 = ws[5];
        double w7 = ws[6];
        double w8 = ws[7];
        double b1 = ws[8];
        double b2 = ws[9];

        double zh1 = w1*x1 + w3*x2 + w7*x3 + b1*1.0;
        double zh2 = w2*x1 + w4*x2 + w8*x3 + b1*1.0;
        double h1 = sigmoid(zh1);
        double h2 = sigmoid(zh2);
        double zy1 = w5*h1 + w6*h2 + b2*1.0;
        double y1 = sigmoid(zy1);

        return y1;
    }

    /**
     * Calculates the fitness of the interneuron weights based on batch learning.
     * @param ws double[] - the interneuron weights
     * @return double
     */
    public double getFitness(double[] ws) {
        double sumErrors = 0;
        double errorSquared = 0;

        for(int i = 0; i < XOR_INPUTS.length; i++) {
            double x1 = XOR_INPUTS[i][0];
            double x2 = XOR_INPUTS[i][1];
            double x3 = XOR_INPUTS[i][2];
            double y1 = feedforward(x1, x2, x3, ws);
            double t1 = XOR_IDEALS[i][0];
            errorSquared = (y1 - t1) * (y1 - t1);
            sumErrors += errorSquared;
        }

        double RMSE = Math.sqrt(sumErrors / XOR_INPUTS.length);

        return RMSE;
    }

    public static void main(final String args[]) {
        //For each weight, get a random weight value
        double[] ws = new double[NUM_WEIGHTS];

        for(int k=0; k < ws.length; k++) {
            ws[k] = getRandomWeight();
        }

        //Print out the weights
        System.out.printf("%1s %9s \n", "Wt", "Value");

        for(int k=0; k < ws.length-2; k++) {
            System.out.printf("%1s %10.4f \n", ("w" + (k + 1)), ws[k]);
        }
        System.out.printf("%1s %9.4f \n", "b1 ", ws[6]);
        System.out.printf("%1s %9.4f \n", "b2 ", ws[7]);

        //Instantiate an instance of XorObjective
        XorObjective objective = new XorObjective();

        System.out.printf("%1s %2s %6s %6s %6s %6s \n", "#", "x1", "x2", "x3", "t1", "y1");

        for(int i = 0; i < XOR_INPUTS.length; i++){
            double y1 = objective.feedforward(XOR_INPUTS[i][0], XOR_INPUTS[i][1], XOR_INPUTS[i][2], ws);
            System.out.printf("%1d %1.4f %1.4f %1.4f %1.4f %1.4f \n", i+1, XOR_INPUTS[i][0], XOR_INPUTS[i][1], XOR_INPUTS[i][2], XOR_IDEALS[i][0], y1);
        }

        double fitness = objective.getFitness(ws);
        System.out.println("fitness = " + fitness);
    }
}