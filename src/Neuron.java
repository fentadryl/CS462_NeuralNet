package myneuralnet;
import java.util.*;

//Author: Ben Rolfe
public class Neuron {
	
    Random random = new Random();
    
    public double weight1 = random.nextDouble(-1, 1); 
    public double weight2 = random.nextDouble(-1, 1);
    public double bias = random.nextDouble(0.05, 0.1);
    
    private double lastInput1;
    private double lastInput2;
    private double lastOutput;
    private double z;
    
    // This is for Forward Propagation
    public double forward(double i1, double i2) {
        this.lastInput1 = i1;
        this.lastInput2 = i2;
        this.z = (weight1 * i1) + (weight2 * i2) + bias;
        this.lastOutput = Util.activation(z);
        return lastOutput;
    }

    // This is for Backpropagation
    public double backward(double errorSignal, double learningRate) {
        // 1. Calculate the local gradient
    	double slope = Util.activationDeriv(lastOutput);
        double delta = errorSignal * slope;

        // 2. Update parameters (using weight1/weight2 to match declarations)
        this.weight1 -= learningRate * delta * lastInput1;
        this.weight2 -= learningRate * delta * lastInput2;
        this.bias -= learningRate * delta;

        // 3. Return the error to the previous layer
        return delta;
    }
}
