package javaai.aann;

import java.util.HashMap;
import java.util.Set;

/**
 * The NNSMap program
 *
 * @author  Timothy Hoang
 * @version 1.0
 * @since   2020-09-03
 */
public class NNSMap extends HashMap<Measure, Species> {
    /**
     * This method is used to train the machines hashmaps.
     * show the usage of various javadoc Tags.
     * @param dest This parameter is the measure for which we are trying to find the nearest neighbor.
     * @return prediction This is the nearest species to the target measure.
     */
    @Override
    public Species get(Object dest) {
        // Starting minimum distance -- the maximum possible value!
        double minDist = Double.MAX_VALUE;
        // Arbitrarily choose a nearest measure
        Set<Measure> keys = this.keySet();
        Measure nearest = (Measure) keys.toArray()[0];
        // Search each measure in the hashmap
        for(Measure src: keys) {
            // Pass 1: get the distance from this src to dest measure
            double dist = getDistance(src,(Measure)dest);
            // If weâ€™re closer than before, update the nearest
            if(dist < minDist) {
                minDist = dist;
                nearest = src;
            }
        }
        // Pass 2: get the species prediction for the nearest one
        Species prediction = super.get(nearest);
        return prediction;
    }

    /**
     * This method is used to get the Euclidean distance between two measures.
     * @param src This parameter is the source measure.
     * @param dest This parameter is the destination measure.
     * @return dist2 This double is the Euclidean distance.
     */
    protected double getDistance(Measure src, Measure dest) {
        // This is the accumulator
        double dist2 = 0;
        // The sepal & petal values are in a 4D array.
        for (int k = 0; k < src.values.length; k++) {
            // Get the difference or delta
            double delta = src.values[k] - dest.values[k];
            // Sum the square differences
            dist2 += (delta * delta);
        }
        // The metric is the sum of square differences.
        // The square root of dist2, is the Euclidean distance.
        return dist2;
    }
}
