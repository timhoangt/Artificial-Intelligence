package javaai.aann;

import java.util.*;

/**
 * The ML2 program
 *
 * @author  Timothy Hoang
 * @version 1.0
 * @since   2020-09-03
 */
public class ML2 extends ML1 {
    /**
     * This is the main method which makes use of the get method in NNSMap as map and the load method in ML1.
     * @param args Unused.
     * @return Nothing.
     */
    public static void main(String[] args) {
        NNSMap map = new NNSMap();
        load(map);
        List<Measure> tests = new ArrayList<>(Arrays.asList(new Measure(5.1, 3.5, 1.4, 0.2),
                new Measure(16.0, 4.2, 3.8, 11.0),
                new Measure(4.2, 0.8, 0.0, 2.0),
                new Measure(5.8,2.7,4.2,1.3),
                new Measure(11.0, 16.0, 4.2, 3.8),
                new Measure(5.1,3.6,1.5,0.4),
                new Measure(0.0,0.0, 0.0, 0.0),
                new Measure(100.0,100.0, 100.0, 100.0),
                new Measure(-1.0, -1.0, -1.0, -1.0)));
        for(Measure measure: tests) {
            System.out.println(map.get(measure) + " " + measure);
        }
    }
}