//Authors: Jaine Tiu, Ben Rolfe
package myneuralnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Util {

	//Author: Ben Rolfe
	// Sigmoid
	public static double sigmoid(double in){
		return 1 / (1 + Math.exp(-in));
	}
	public static double sigmoidDeriv(double output){
    	return output * (1 - output);
	}

	//Author: Jaine Tiu
	// ReLU
    public static double relu(double in){
        return Math.max(0, in);
    }
    public static double reluDeriv(double in){
        return in > 0 ? 1 : 0;
    }

	//Author: Jaine Tiu
	//Tanh
	public static double tanh(double in){
    	return Math.tanh(in);
    }  
    public static double tanhDeriv(double in){
    	double t = Math.tanh(in);
        return 1 - t * t;  
    }

	//Author: Jaine Tiu
	public static double activation(double in){
		return sigmoid(in);
        // return relu(in);
		// return tanh(in);
    }
    public static double activationDeriv(double in){
        return sigmoidDeriv(in);
        // return reluDeriv(in);
		// return tanhDeriv(in);
    }

	//Author: Ben Rolfe
	// Assumes array args are same length
	public static Double meanSquareLoss(List<Double> correctAnswers, List<Double> predictedAnswers){
	    double sumSquare = 0;
	    for (int i = 0; i < correctAnswers.size(); i++){
	        double error = correctAnswers.get(i) - predictedAnswers.get(i);
			sumSquare += (error * error);
	    }
	    return sumSquare / (correctAnswers.size());   
	}
	
	//Author: Ben Rolfe
	//Methods used for normalization
	public static double normalize(double val, double min, double max) {
	    return (val - min) / (max - min);
	} 
	public static List<List<Double>> getNormalizedData(List<List<Integer>> raw) {
    	List<List<Double>> normalizedData = new ArrayList<>();

    	double[] wMinMax = getMinMax(raw, 0);
    	double[] hMinMax = getMinMax(raw, 1);
    	double minW = wMinMax[0], maxW = wMinMax[1];
    	double minH = hMinMax[0], maxH = hMinMax[1];

    	for (int i = 0; i < raw.size(); i++) {
        	double nW = normalize(raw.get(i).get(0), minW, maxW);
        	double nH = normalize(raw.get(i).get(1), minH, maxH);
        	normalizedData.add(Arrays.asList(nW, nH));
    	}
    	return normalizedData;
	}
	public static double[] getMinMax(List<List<Integer>> data, int index) {
    	double min = Double.MAX_VALUE;
    	double max = -Double.MAX_VALUE;

    	for (int i = 0; i < data.size(); i++) {
        	double val = data.get(i).get(index);

        	if (val < min) {
				min = val;
			}
        	if (val > max) {
				max = val;
			}
    	}
    	return new double[]{min, max};
	}
}
