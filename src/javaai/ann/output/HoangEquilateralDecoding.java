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
package javaai.ann.output;

import javaai.util.Helper;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * This class tests equilateral decoding tolerance for the iris data set.
 * @author Timothy Hoang
 */
public class HoangEquilateralDecoding {
    /** Number of tests to run */
    public final static int NUM_TESTS = 100;

    /** Tolerance as a percent, e.g., 1.0 == 1% */
    public final static double TOLERANCE = 100.0;

    /** Copy in the values from running EquilateralEncoding */
    static double ideals[][] = {
            {-0.8660, -0.5000},   // Viginica
            {0.8660, -0.5000},   // Setosa
            {0.0000, 1.0000}    // Versicolor
    };

    /** Specie names -- order MUST correspond to measures */
    static final List<String> species =
            new ArrayList<>(Arrays.asList("viginica", "setosa", "versicolor"));

    public static int count = 1;

    /**
     * Launch point for program.
     * @param args Command line arguments.
     */
    public static void main(String[] args) {
        System.out.printf("%-4s %-10s %-18s %-1s %-10s %s %n", "#", "ideal", "Encoding", "|", "actual", "Encoding");
        EquilateralEncoding.load();

        Random ran = new Random( );

        int success = 0;

        for(int n=0; n < NUM_TESTS; n++) {
            /** Pick a species randomly */
            int idealIndex = ran.nextInt(ideals.length);

            /** 1. Get a random encoding from ideals using idealIndex.*/
            double[] encodings = ideals[idealIndex];
            /** 2. Create a new array of activations perturbed by the tolerance divided by 100.*/
            double[] activations = new double[encodings.length];

            for(int k=0; k < encodings.length; k++) {
                double epsilon = 1 + ran.nextGaussian() * TOLERANCE / 100.0;
                activations[k] = encodings[k] * epsilon;
            }
            /** 3. Decode these perturbed activations.*/
            int actualIndex = EquilateralEncoding.eq.decode(activations);
            /** 4. If the predicted index equals the actual index, update success count.*/
            System.out.printf("%-4d %-10s %-18s %-1s %-10s %s", count, species.get(idealIndex), Helper.asString(ideals[idealIndex]), "|", species.get(actualIndex), Helper.asString(ideals[actualIndex]));
            count++;
            if(idealIndex == actualIndex) {
                success++;
                System.out.println();
            }
            else{
                System.out.println(" MISSED!");
            }

        }

        double rate = (double)success / NUM_TESTS * 100;

        System.out.printf("accuracy = %d of %d or %.0f%%, perturbance = %.0f%%\n",success, NUM_TESTS, rate, TOLERANCE);
    }


}
