package myneuralnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class Util {
	  public static double sigmoid(double in){
	    return 1 / (1 + Math.exp(-in));
	  }
	  public static double sigmoidDeriv(double in){
	      double sigmoid = Util.sigmoid(in);
	      return sigmoid * (1 - in);
	    }
	    /** Assumes array args are same length */
	    public static Double meanSquareLoss(List<Double> correctAnswers, List<Double> predictedAnswers){
	      double sumSquare = 0;
	      for (int i = 0; i < correctAnswers.size(); i++){
	        double error = correctAnswers.get(i) - predictedAnswers.get(i);
		sumSquare += (error * error);
	      }
	      return sumSquare / (correctAnswers.size());   
	}
	    
	    public static double normalize(double val, double min, double max) {
	        return (val - min) / (max - min);
	    }
	    
	    public static List<List<Double>> getNormalizedData(List<List<Integer>> raw, double minW, double maxW, double minH, double maxH) {
	        List<List<Double>> normalizedData = new ArrayList<>();
	        for (List<Integer> row : raw) {
	            double nW = normalize(row.get(0), minW, maxW);
	            double nH = normalize(row.get(1), minH, maxH);
	            normalizedData.add(Arrays.asList(nW, nH));
	        }
	        return normalizedData;
	    }
}