package javaai.ann.basic;

import javaai.ann.output.Ontology;
import javaai.util.Helper;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.mathutil.Equilateral;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * XOR: This example is essentially the "Hello World" of neural network
 * programming. This example shows how to construct an Encog neural network to
 * predict the output from the XOR operator. This example uses backpropagation
 * to train the neural network.
 *
 * This example attempts to use a minimum of Encog values to create and train
 * the neural network. This allows you to see exactly what is going on. For a
 * more advanced example, that uses Encog factories, refer to the XORFactory
 * example.
 *
 * The original version of this code does not appear to converge. I fixed this
 * problem by using two neurons in the hidden layer and instead of ramped activation,
 * sigmoid activation. This makes the network reflect the model in figure 1.1
 * in the book, p. 11. I also added more comments to make the code more
 * explanatory.
 * @author Timothy Hoang
 * @date 20 Oct 2020
 */
public class HoangzIris {
    /** Error tolerance */
    public final static double TOLERANCE = 0.01;

    /** The high range index */
    public final static int HI = 1;

    /** The low range index */
    public final static int LO = 0;

    /** How much of the data will be used for training purposes */
    public static final double TRAINING_PERCENT = 0.80;

    /**
     * The input necessary for XOR.
     */
    public static double XOR_INPUTS[][] = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
    };

    /**
     * The ideal data necessary for XOR.
     */
    public static double XOR_IDEALS[][] = {
            {0.0},
            {1.0},
            {1.0},
            {0.0}
    };

    protected static double TRAINING_INPUTS [][];

    protected static double TRAINING_IDEALS [][];

    protected static double TESTING_INPUTS [][];

    protected static double TESTING_IDEALS [][];

    public static Equilateral eq;

    public static List<String> subtypes;

    /**
     * Gets hi-lo range using an elementary form of unsupervised learning.
     * @param list List
     * @return 2-tuple of doubles for low and high range
     */
    protected static double[] getRange(List<Double> list) {
        // Initial low and high values
        double[] range = {Double.MAX_VALUE, -Double.MAX_VALUE};

        // Go through each value in the list
        for(Double value: list) {
            if(value > range[HI]){
                range[HI] = value;
            }

            if(value < range[LO]){
                range[LO] = value;
            }
        }

        return range;
    }

    /**
     * Returns the normalized inputs in the required form.
     * @return Double matrix containing normalized input values.
     */
    protected double[][] getInputs(){

        HashMap<String, List<Double>> normals = new HashMap<>( );

        for(String title : Helper.headers) {

            List<Double> col = (List<Double>) Helper.data.get(title);

            //If the data list is empty, null, or not of Double ontology, skip it.
            if(col.isEmpty() || col == null || col.get(0) instanceof Double == false) {
                continue;
            }

            //Get the range of the data instance
            double[] range = getRange(col);

            NormalizedField norm = new NormalizedField(NormalizationAction.Normalize,null, range[HI], range[LO], 1, -1);

            //Contains normalized data
            List<Double> normalized = new ArrayList<>();

            for(int i = 0; i < col.size(); i++) {
                double normalizedValue = norm.normalize(col.get(i));
                normalized.add(normalizedValue);
            }

            //Add normalized data to the normals for this title
            normals.put(title, normalized);

        }

        //keySet returns an Object array, not a String array
        Object[] keys = normals.keySet().toArray();

        //Transfer header names to String array--we use it later
        String[] cols = new String[keys.length];
        System.arraycopy(keys, 0, cols, 0, keys.length);

        //Allocate the 2D storage
        int numRows = Helper.getRowCount();
        int numCols = cols.length;
        double[][] inputs = new double[numRows][numCols];

        //Transfer the data from normals to inputs
        for(int row = 0; row < numRows; row++) {
            for(int col = 0; col < numCols; col++) {
                String title = cols[col];
                inputs[row][col] = normals.get(title).get(row);
            }
        }

        return inputs;

    }

    protected double[][] getIdeals() throws Exception {

        //get the subtypes
        final int SPECIES = 4;

        subtypes = Helper.getNominalSubtypes(SPECIES);

        double[][] ideals = new double[Helper.getRowCount()][];

        HashMap<String,Integer> subtypeToNumber = new HashMap<>();
        Integer number = 0;
        for(String subtype: subtypes) {
            subtypeToNumber.put(subtype,number);
            number++;
        }

        eq = new Equilateral(subtypes.size(), 1.0, -1.0);

        String col = Helper.headers.get(SPECIES);

        for(int rowno=0; rowno < Helper.getRowCount(); rowno++) {
            // Get the nominal as a string name
            String nominal = (String) Helper.data.get(col).get(rowno);
            // Convert the name to a subspecies index number
            number = subtypeToNumber.get(nominal);
            if(number == null)
                throw new Exception("bad nominal: "+nominal);
            // Encode the number as vertex in n-1 dimensions
            double[] encoding = eq.encode(number);
            // Save the vertex encoding as columns for this row
            ideals[rowno] = encoding;
        }

        /*
        System.out.println("Iris encoded data outputs" +
                "\n-------------------------" +
                "\nIndex      y1            y2     Decoding");
        for(int row = 0; row < ideals.length; row++) {
            System.out.println();
            int index = row+1;
            System.out.print("  " + index );
            for(int column = 0; column < ideals[0].length; column++) {
                System.out.printf("%12.4f ", ideals[row][column]);
            }
            System.out.print("   " + Helper.data.get(col).get(eq.decode(ideals[row])));
        }
        System.out.println();
         */

        return ideals;

    }

    //Index beginnings and ends for the training and testing sets
    public static final int getTrainingStartIndex() {
        return 0;
    }

    public static final int getTrainingEndIndex() {
        return (int) (Helper.getRowCount() * TRAINING_PERCENT + 0.50 -1);
    }

    public static final int getTestingStartIndex() {
        return (int) (Helper.getRowCount() * TRAINING_PERCENT + 0.50);
    }

    public static final int getTestingEndIndex() {
        return Helper.getRowCount() - 1;
    }

    /**
     * Loads in the CSV, gets the normalized input, and outputs the normalized values.
     */
    protected static void init() {
        try {
            Helper.loadCsv("iris.csv", Ontology.parsers);
        } catch (Exception e) {
            e.printStackTrace();
        }

        HoangzIris iris = new HoangzIris();

        double[][] inputs = iris.getInputs();

        /*
        System.out.println("Iris normalized data inputs" + "\n-----------------------------------------------------" +
                "\n Index  Sepal Length  Sepal Width  Petal Length  Petal Width");

        for(int row = 0; row < inputs.length; row++) {
            System.out.println();
            int index = row+1;
            System.out.print("  " + index );
            for(int col = 0; col < inputs[0].length; col++) {
                System.out.printf("%12.2f ", inputs[row][col]);
            }
        }
        System.out.println();
         */

        double[][] ideals = new double[Helper.getRowCount()][];

        try {
            ideals = iris.getIdeals();
        } catch (Exception e) {
            e.printStackTrace();
        }

        //Initialize training sets and proves program correctness for 120 rows
        int numRowsTraining = getTrainingEndIndex() - getTrainingStartIndex() + 1;
        assert(numRowsTraining == 120);

        int numRowsTesting = getTestingEndIndex() - getTestingStartIndex() + 1;


        //Create training sets
        TRAINING_INPUTS = new double[numRowsTraining][];

        TRAINING_IDEALS = new double[numRowsTraining][];

        System.arraycopy(inputs, 0, TRAINING_INPUTS,0, TRAINING_INPUTS.length);

        System.arraycopy(ideals, 0, TRAINING_IDEALS,0, TRAINING_IDEALS.length);

        TESTING_INPUTS = new double[numRowsTesting][];

        TESTING_IDEALS = new double[numRowsTesting][];

        System.arraycopy(inputs, getTestingStartIndex(), TESTING_INPUTS,0, TESTING_INPUTS.length);

        System.arraycopy(ideals, getTestingStartIndex(), TESTING_IDEALS,0, TESTING_IDEALS.length);

        /*
        //Printing the training inputs and ideals for testing
        System.out.println("TRAINING INPUTS" + "\n-----------------------------------------------------" +
                "\n Index Sepal Length Sepal Width Petal Length Petal Width");
        for(int row = 0; row < TRAINING_INPUTS.length; row++) {
            System.out.println();
            int index = row+1;
            System.out.print("  " + index );
            for(int col = 0; col < TRAINING_INPUTS[0].length; col++) {
                System.out.printf("%12.2f ", TRAINING_INPUTS[row][col]);
            }
        }
        System.out.println();
         */

        /*
        System.out.println("TRAINING IDEALS" + "\n-------------------------" +
                "\nIndex      y1            y2");
        for(int row = 0; row < TRAINING_IDEALS.length; row++) {
            System.out.println();
            int index = row+1;
            System.out.print("  " + index );
            for(int col = 0; col < TRAINING_IDEALS[0].length; col++) {
                System.out.printf("%12.2f ", TRAINING_IDEALS[row][col]);
            }
        }
        System.out.println();
        */
    }

    /**
     * The main method.
     *
     * @param args No arguments are used.
     */
    public static void main(final String args[]) {

        init();

        // Create a neural network, without using a factory
        BasicNetwork network = new BasicNetwork();

        // Add input layer with no activation function, bias enabled, and two neurons
        network.addLayer(new BasicLayer(null, true, 4));

        // Add hidden layer with ramped activation, bias enabled, and five neurons
        // NOTE: ActivationReLU is not in javadoc but can be found here http://bit.ly/2zyxk7A.
        // network.addLayer(new BasicLayer(new ActivationReLU(), true, 5));

        // Add hidden layer with sigmoid activation, bias enabled, and two neurons
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 4));

        // Add output layer with sigmoid activation, bias disable, and one neuron
        network.addLayer(new BasicLayer(new ActivationTANH(), false, 2));

        // No more layers to be added
        network.getStructure().finalizeStructure();

        // Randomize the weights
        network.reset();

        System.out.println("Network description: before training");
        Helper.describe(network);

        // Creates the training data
        MLDataSet trainingSet = new BasicMLDataSet(TRAINING_INPUTS, TRAINING_IDEALS);

        // Train the neural network.
        // Use a training object to train the network, in this case, an improvement
        // back propagation. For details on what this does see the javadoc.
        final ResilientPropagation train = new ResilientPropagation(network, trainingSet);

        int epoch = 1;

        System.out.println("   epoch  error");
        Helper.log(epoch, train,false);
        do {
            train.iteration();

            epoch++;

            Helper.log(epoch, train,false);

        } while (train.getError() > TOLERANCE && epoch < Helper.MAX_EPOCHS);

        train.finishTraining();

        Helper.log(epoch, train,true);
        System.out.println("Network training results:");
        Helper.report(trainingSet, network);
        System.out.println("Network description: after training");
        Helper.describe(network);

        Encog.getInstance().shutdown();

        int missed = 0;
        double[] output = new double[2];

        int outputIndex;
        int idealIndex;
        int index = 1;

        String outputSubtype;
        String idealSubtype;

        System.out.print("Network testing results:");
        for(int k=0; k < TESTING_INPUTS.length; k++) {
            double[] input = TESTING_INPUTS[k];

        //Send input into MLP and get its output
            network.compute(input, output);

        //Decode output to its actual subtype index.
            outputIndex = eq.decode(output);

        //Decode ideal to its subtype index.
            idealIndex = eq.decode(TESTING_IDEALS[k]);

        //Output actual & ideal specie string names.
            outputSubtype = subtypes.get(outputIndex);
            idealSubtype = subtypes.get(idealIndex);

        //If actual != ideal, output MISSED! in the right margin
            System.out.printf("\n%2s %12s %12s", index, idealSubtype, outputSubtype);
            index++;

            if(outputIndex != idealIndex) {
                System.out.print(" MISSED!");
                missed++;
            }
        //If actual != ideal, increment missed.
        }

        double successRate = ((double) TESTING_IDEALS.length - missed) / TESTING_IDEALS.length * 100;
        System.out.printf("\nSuccess Rate = " + (TESTING_IDEALS.length - missed) + "/"
                + TESTING_IDEALS.length + " (%3.1f" + "%%)", successRate);

        Encog.getInstance().shutdown();

        /*
        // Never train on a specific error rate but an acceptable tolerance and
        // if the error drops below that tolerance, the network has converged.
        do {
            long then = System.nanoTime();

            train.iteration();

            long now = System.nanoTime();

            long elapsed = now - then;

            System.out.println("dt: "+elapsed+ " epoch #" + epoch + " error: " + train.getError());

            epoch++;
        } while (train.getError() > TOLERANCE);

        train.finishTraining();

        // Test the neural network
        System.out.println("Neural Network Results:");

        for (MLDataPair pair : trainingSet) {

            final MLData output = network.compute(pair.getInput());

            System.out.println(pair.getInput().getData(0) + "," + pair.getInput().getData(1)
                    + ", actual=" + output.getData(0) + ",ideal=" + pair.getIdeal().getData(0));
        }

        Encog.getInstance().shutdown();

         */
    }
}
