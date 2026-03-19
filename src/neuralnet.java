package myneuralnet;
import java.util.*;

public class neuralnet {

	public static void main(String[] args) {
		
		List<List<Integer>> data = new ArrayList<List<Integer>>();
		data.add(Arrays.asList(115, 66));
		data.add(Arrays.asList(175, 78));
		data.add(Arrays.asList(205, 72));
		data.add(Arrays.asList(120, 67));
		List<Double> answers = Arrays.asList(1.0,0.0,0.0,1.0); 
		
		
		// 2. APPLY NORMALIZATION (Creating the separate named list)
	    // We use 100-250 for weight and 60-85 for height as our "bounds"
	    List<List<Double>> normalizedData = Util.getNormalizedData(data, 100, 250, 60, 85);
		
		int epochs = 2000;
	    double lr = 0.1;
	    Network network = new Network(epochs);
		
		System.out.println("Training in progress...");
	    
	    // Capture the loss history from the train method
	    List<Double> history = network.train(normalizedData, answers, lr);

	    // Now we print the stats here in main!
	    System.out.println("\n--- Training Stats ---");
	    for (int i = 0; i < history.size(); i++) {
	        // Print every 200 epochs to keep the console clean
	        if (i % 200 == 0 || i == history.size() - 1) {
	            System.out.printf("Epoch %d | Loss: %.6f%n", i, history.get(i));
	        }
	    }

	    System.out.println("\n--- Final Predictions ---");
	    for (int i = 0; i < normalizedData.size(); i++) {
	        double p = network.predict(normalizedData.get(i).get(0), normalizedData.get(i).get(1));
	        System.out.printf("Target: %.1f | Prediction: %.4f%n", answers.get(i), p);
	    }
	}
		
		
}

