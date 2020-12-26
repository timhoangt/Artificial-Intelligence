package javaai.aann;

import java.util.*;

/**
 * The ML3 program
 *
 * @author  Timothy Hoang
 * @version 1.0
 * @since   2020-09-11
 */

public class ML3 extends ML2 {

    public final static double TRAINING_PERCENT = 0.80;

    /**
     * This is the main method which makes use of the get method in NNSMap as nns and the load method in ML1.
     * @param args Unused.
     * @return Nothing.
     */
    public static void main(String[] args) {
        Map<Measure, Species> ideals = new HashMap<>();
        load(ideals);
        measures.add(new Measure());

        int testingStart = (int) (ideals.size() * TRAINING_PERCENT);

        NNSMap nns = new NNSMap();
        for(int k=0; k < testingStart; k++) {
            nns.put(measures.get(k), flowers.get(k));
        }

        int tried = 0;
        int missed = 0;
        int number  = 0;

        for(int k = testingStart; k < measures.size(); k++) {
            number++;
            Measure test = measures.get(k);
            tried++;
            //runs test through ideals: call result â€˜idealâ€™
            Species ideal = ideals.get(test);

            //runs test through NNS: call result â€˜actualâ€™
            Species actual = nns.get(test);

            //compare actual & ideal, and if they differ, report discrepancy and increment missed
            if (!actual.equals(ideal)) {
                missed++;
                System.out.println(number + " " + test + " " + ideals.get(test) + " " + nns.get(test) + " MISSED!");
            }
            else {
                System.out.println(number + " " + test + " " + ideals.get(test) + " " + nns.get(test));
            }
        }
        int hit = tried-missed;
        int percentage = hit * 100 / tried;
        System.out.println("accuracy: " + hit + " of " + tried + " or " + percentage + "%");
    }

}