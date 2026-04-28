package myneuralnet;
import java.util.*;

//Author: Ben Rolfe
//Constructor for a Neuron has weight, memory, etc for forward/backward propagation
public class Neuron {
    
    Random random = new Random();
    
    public double weight1 = random.nextDouble(-1, 1); 
    public double weight2 = random.nextDouble(-1, 1);
    public double bias = random.nextDouble(0.05, 0.1);
    
    private double lastInput1;
    private double lastInput2;
    private double lastOutput;
    private double z;
    private boolean outputNeuron;
    
    public Neuron() {
        this(false);
    }

    //flag for the output neuron
    //Output must always use sigmoid activation to produce proper results for rELU
    public Neuron(boolean outputNeuron) {
        this.outputNeuron = outputNeuron;
    }
    
    // This is for Forward Propagation
    public double forward(double i1, double i2) {
        this.lastInput1 = i1;
        this.lastInput2 = i2;
        this.z = (weight1 * i1) + (weight2 * i2) + bias;

        if (outputNeuron) {
            // Final node always outputs sigmoid, so predictions stay between 0 and 1.
            this.lastOutput = Util.sigmoid(z);
        } else {
            // Hidden nodes use the selected hidden activation from Util.activation().
            this.lastOutput = Util.activation(z);
        }
        return lastOutput;
    }

    // This is for Backpropagation
    public double backward(double errorSignal, double learningRate) {
        // 1. Calculate the local gradient
        double slope;

        if (outputNeuron) {
            // Sigmoid derivative using the output of sigmoid(z).
            slope = lastOutput * (1 - lastOutput);
        } else {
            // Hidden activation derivative should use z, not lastOutput.
            slope = Util.activationDeriv(z);
        }

        double delta = errorSignal * slope;

        // 2. Update parameters
        this.weight1 -= learningRate * delta * lastInput1;
        this.weight2 -= learningRate * delta * lastInput2;
        this.bias -= learningRate * delta;

        // 3. Return the error to the previous layer
        return delta;
    }
}
