import java.util.Arrays;

public class Perceptron {
    private double[] weights;
    private double bias;

    public Perceptron(int num_inputs) {
        weights = new double[num_inputs];
        for (int i = 0; i < num_inputs; i++) {
            weights[i] = Math.random();
        }
        bias = Math.random();
    }

    public int predict(double[] inputs) {
        double weighted_sum = bias;
        for (int i = 0; i < inputs.length; i++) {
            weighted_sum += weights[i] * inputs[i];
        }
        return activate(weighted_sum);
    }

    private int activate(double x) {
        return x > 0 ? 1 : 0;
    }

    public static void main(String[] args) {
        Perceptron p = new Perceptron(2);
        double[][] training_data_inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] training_data_labels = {0, 0, 0, 1};
        double learning_rate = 0.1;
        int num_epochs = 10;
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            for (int i = 0; i < training_data_inputs.length; i++) {
                double[] inputs = training_data_inputs[i];
                int label = training_data_labels[i];
                int prediction = p.predict(inputs);
                int error = label - prediction;
                p.bias += learning_rate * error;
                for (int j = 0; j < inputs.length; j++) {
                    p.weights[j] += learning_rate * error * inputs[j];
                }
            }
        }
        System.out.println("0 0: " + p.predict(new double[]{0, 0}));
        System.out.println("0 1: " + p.predict(new double[]{0, 1}));
        System.out.println("1 0: " + p.predict(new double[]{1, 0}));
        System.out.println("1 1: " + p.predict(new double[]{1, 1}));
    }
}
