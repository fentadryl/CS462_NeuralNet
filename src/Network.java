//Author: Jaine Tiu
package myneuralnet;
import java.util.*;

public class Network {
    
    int epochs = 0; //1000;
    Double learnFactor = null;
    List<Neuron> neurons = Arrays.asList(
        new Neuron(), new Neuron(), new Neuron(), // input nodes use hidden activation
        new Neuron(), new Neuron(),               // hidden nodes use hidden activation
        new Neuron(true));                        // output node always uses sigmoid
    
    public Network(int epochs) {
        this.epochs = epochs;
    }
    
    public Network(int epochs, Double learnFactor) {
        this.epochs = epochs;
        this.learnFactor = learnFactor;
    }
      
    public List<Double> train(List<List<Double>> data, List<Double> answers, double learningRate) {
          
        List<Double> lossHistory = new ArrayList<>();
          
        for (int epoch = 0; epoch < this.epochs; epoch++) {
            double epochLoss = 0;

            for (int i = 0; i < data.size(); i++) {
                double target = answers.get(i);
                double weight = data.get(i).get(0);
                double height = data.get(i).get(1);

                // Calculate loss before we update the weights
                double prediction = this.predict(weight, height);
                epochLoss += Math.pow(prediction - target, 2);

                this.trainStep(weight, height, target, learningRate);
            }
              
            // Store the average loss for this epoch
            lossHistory.add(epochLoss / data.size());
        }
        return lossHistory;
    }
      
    // This is your Forward Pass "Map"
    public double predict(double input1, double input2) {
        double n0 = neurons.get(0).forward(input1, input2);
        double n1 = neurons.get(1).forward(input1, input2);
        double n2 = neurons.get(2).forward(input1, input2);
          
        double n3 = neurons.get(3).forward(n1, n0);
        double n4 = neurons.get(4).forward(n2, n1);
          
        // Neuron 5 was constructed as an output neuron, so its forward()
        // always uses sigmoid instead of the hidden activation.
        return neurons.get(5).forward(n4, n3);
    }

    public void trainStep(double input1, double input2, double target, double learningRate) {
        // 1. Forward Pass (Calculates and Saves Memory)
        double prediction = this.predict(input1, input2);

        // 2. Calculate Error (Direction and Magnitude)
        double initialError = 2 * (prediction - target);

        // 3. Backward Pass (The Chain Rule in reverse)
        // Neuron 5 uses sigmoid derivative because it is the output neuron.
        double errorFromOutput = neurons.get(5).backward(initialError, learningRate);
          
        double errorToN4 = neurons.get(4).backward(errorFromOutput, learningRate);
        double errorToN3 = neurons.get(3).backward(errorFromOutput, learningRate);
          
        neurons.get(2).backward(errorToN4, learningRate);
        neurons.get(1).backward((errorToN4 + errorToN3) / 2, learningRate); 
        neurons.get(0).backward(errorToN3, learningRate);
    }
}
