package javaai.aann;

import java.util.*;

/**
 * The ML1 program
 *
 * @author  Timothy Hoang
 * @version 1.0
 * @since   2020-08-31
 */
public class ML1 extends BaseML {
    /**
     * This is the main method which makes use of load method.
     * This method
     * @param args Unused.
     * @return Nothing.
     */
    public static void main(String[] args) {
        Map<Measure,Species> map = new HashMap<>();

        load(map);

        List<Measure> tests = new ArrayList<>(Arrays.asList(measures.get(93), measures.get(56), measures.get(70), measures.get(35)));

        for(Measure measure: tests) {
            System.out.println(map.get(measure) + " " + measure);
        }
    }

    /**
     * This method is used to train the machines hashmaps.
     * show the usage of various javadoc Tags.
     * @param target This parameter is an instance of Map<Measure,Species>.
     * @return Nothing.
     */
    public static void load(Map<Measure,Species> target) {
        BaseML.load();

        for(int k=0; k < measures.size(); k++) {
            target.put(measures.get(k), flowers.get(k));
        }
    }
}
