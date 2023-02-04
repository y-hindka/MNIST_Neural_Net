package src;
//////////// FILE HEADER ///////////////
// Author: Yash Hindka (yhindka@wisc.edu)
// CS540, Young Wu
// Lecture 002, Summer 2020
// Project: P1 Logistic Regression/Neural Network on MNIST Dataset
// 6/7/2020 ////////////////////////////

import java.util.ArrayList;
import java.util.Random;

public class NeuralNetwork {

  private double[][] L1weights;
  private double[] L1bias;
  private double[] L2weights;
  private double learningRate;
  private ArrayList<Integer> labels;
  private ArrayList<ArrayList<Double>> trainingData;
  private int numberHiddenUnits;
  private ArrayList<Double> costs;
  private int batchSize;

  public NeuralNetwork(ArrayList<ArrayList<Double>> trainingData, ArrayList<Integer> labels) {

    this.trainingData = trainingData;
    this.labels = labels;
    this.numberHiddenUnits = trainingData.get(0).size() / 2;
    this.L1weights = new double[trainingData.get(0).size()][numberHiddenUnits]; // 784 arrays of
                                                                                // length 392
    // initialize L1weights with random values between -1 and 1
    for (int r = 0; r < L1weights.length; r++) {
      for (int c = 0; c < L1weights[r].length; c++) {
        L1weights[r][c] = Math.random() * 2 - 1.0; //Math.random() * 0.2 - 0.1;
      }
    }
    this.L1bias = new double[numberHiddenUnits];
    // initialize bias weights to 0
    for (int i = 0; i < L1bias.length; i++) {
      L1bias[i] = Math.random() * 2 - 1.0; //Math.random() * 0.2 - 0.1;
    }
    this.L2weights = new double[numberHiddenUnits + 1];
    // initialize random L2 weights
    for (int i = 0; i < L2weights.length; i++) {
      L2weights[i] = Math.random() * 2 - 1.0; //Math.random() * 0.2 - 0.1;
    }

    this.learningRate = 0.1;
    costs = new ArrayList<>();
    this.batchSize = 1;
  }

  public double[][] backpropagation() {
 
    int e = 0;
    while (e < 10) {
      costs.clear();
      shuffle();
      int iteration = 0;
      int batchBegin = 0;
      int batchEnd = batchSize;
      while (iteration < trainingData.size()) {
  
        double[][] hiddena_iArray = evaluateActivationFunctionLayer1(batchBegin, batchEnd);
        double[] output = evaluateActivationFunctionLayer2(hiddena_iArray);
        double[][] meanL1Gradients = new double[trainingData.get(0).size()][numberHiddenUnits];
        double[] meanL1biases = new double[numberHiddenUnits];
        double[] meanL2Gradients = new double[numberHiddenUnits];
        double meanL2biasgradient = 0;
  
        // allows batch gradient descent
        for (int i = 0; i < batchEnd - batchBegin; i++) {
          
          meanL1Gradients = this.getL1Gradient(output[i], batchBegin+i, hiddena_iArray[i], meanL1Gradients);
          meanL1biases = this.getL1biases(meanL1biases, output[i], batchBegin+i, hiddena_iArray[i]);
          meanL2Gradients = this.getL2Gradient(output[i], batchBegin+i, hiddena_iArray[i], meanL2Gradients);
          meanL2biasgradient = this.getL2bias(output[i], batchBegin+i);
        }
        
        updateL1Weights(meanL1Gradients, meanL1biases, batchEnd - batchBegin);
        updateL2Weights(meanL2Gradients, meanL2biasgradient, batchEnd - batchBegin);
  
        costs.add(totalCost(output, batchBegin, batchEnd));
  
        iteration += batchSize;
        batchBegin += batchSize;
        batchEnd += batchSize;
        if (batchEnd > trainingData.size()) { batchEnd = trainingData.size(); }
      }
      // update costs
      double sum = 0;
      for (double c : costs) {
        sum += c;
      }
      System.out.println(e + ": cost: " + sum);
      e++;
    }

    // create array with all weights to return
    // +2 for L1bias and L2weights, +1 for L2weights bias term
    double[][] allWeights = new double[trainingData.get(0).size() + 2][numberHiddenUnits + 1];
    for (int r = 0; r < allWeights.length; r++) {
      for (int c = 0; c < allWeights[r].length; c++) {

        if (r < allWeights.length - 2 && c < allWeights[r].length - 1) {
          allWeights[r][c] = L1weights[r][c];
        } else if (r == allWeights.length - 2 && c < allWeights[r].length - 1) {
          allWeights[r][c] = L1bias[c];
        } else if (r == allWeights.length - 1 && c < allWeights[r].length - 1) {
          allWeights[r][c] = L2weights[c];
        }
      }
    }
    // add l2 bias
    allWeights[allWeights.length - 1][allWeights[0].length - 1] = L2weights[L2weights.length - 1];

    return allWeights;

  }
  
  @SuppressWarnings("unchecked")
  private void shuffle() {
    
    Random r = new Random();
    for (int i = trainingData.size() - 1; i > 0; i--) {
      
      int randomIndex = r.nextInt(i);
      // swap random elements
      Object tempTD = trainingData.get(randomIndex);
      trainingData.set(randomIndex, trainingData.get(i));
      trainingData.set(i, (ArrayList<Double>) tempTD);
      
      Object tempLabel = labels.get(randomIndex);
      labels.set(randomIndex, labels.get(i));
      labels.set(i, (Integer) tempLabel);
    }
  }

  private double[][] evaluateActivationFunctionLayer1(int batchBegin, int batchEnd) {

    double[][] hiddenz_iArray = new double[batchSize][numberHiddenUnits];
    double[][] hiddena_iArray = new double[batchSize][numberHiddenUnits];
    

    // get z_ij's for layer 1
    for (int i = 0; i < batchEnd - batchBegin; i++) {
      for (int k = 0; k < numberHiddenUnits; k++) {
        double sum = 0;
        for (int j = 0; j < trainingData.get(batchBegin+i).size(); j++) {

          sum += L1weights[j][k] * trainingData.get(batchBegin+i).get(j);
        }
        // add bias
        sum += L1bias[k];
        
        hiddenz_iArray[i][k] = sum;
      }
    }

    // get a_ij's for layer 1
    for (int r = 0; r < hiddenz_iArray.length; r++) {
      for (int c = 0; c < hiddenz_iArray[r].length; c++) {
        hiddena_iArray[r][c] = (1 / (1 + Math.pow(Math.E, hiddenz_iArray[r][c] * -1)));
      }
    }

    return hiddena_iArray;
  }

  private double[] evaluateActivationFunctionLayer2(double[][] hiddena_iArray) {

    double[] z_iArray = new double[hiddena_iArray.length];
    double sum;
    // get z_i's for second layer
    for (int i = 0; i < hiddena_iArray.length; i++) {
      sum = 0;
      for (int j = 0; j < hiddena_iArray[i].length; j++) {
        sum += hiddena_iArray[i][j] * L2weights[j];
      }
      // add bias
      sum += L2weights[numberHiddenUnits];
      z_iArray[i] = sum;
    }

    // get a_i's for second layer
    double[] a_iArray = new double[z_iArray.length];
    for (int i = 0; i < z_iArray.length; i++) {

      a_iArray[i] = 1 / (1 + Math.pow(Math.E, z_iArray[i] * -1));
    }

    return a_iArray;
  }

  private void updateL1Weights(double[][] meanL1Gradients, double[] meanL1biases, int currentBatchSize) {

    for (int r = 0; r < L1weights.length; r++) {
      for (int c = 0; c < L1weights[r].length; c++) {

        L1weights[r][c] = L1weights[r][c] - (this.learningRate * (meanL1Gradients[r][c] / currentBatchSize));
      }
    }

    // update bias
    for (int i = 0; i < L1bias.length; i++) {
      L1bias[i] = L1bias[i] - (this.learningRate * (meanL1biases[i] / currentBatchSize));
    }
  }
  
  private double[][] getL1Gradient(double output, int iteration, double[] hiddena_iArray, double[][] meanGradients) {
    
      for (int j = 0; j < trainingData.get(iteration).size(); j++) {
        for (int i = 0; i < numberHiddenUnits; i++) {
          meanGradients[j][i] += ((output - labels.get(iteration)) * output * (1 - output) * L2weights[i] * hiddena_iArray[i] * (1 - hiddena_iArray[i]) 
              * trainingData.get(iteration).get(j));
        }
      }
    
    return meanGradients;
  }
  
  private double[] getL1biases(double[] meanBiases, double output, int iteration, double[] hiddena_iArray) {
    
    // there are 392 biases
    for (int i = 0; i < numberHiddenUnits; i++) {
      meanBiases[i] += ((output - labels.get(iteration)) * output * (1 - output) * L2weights[i] * hiddena_iArray[i] * (1 - hiddena_iArray[i]));
    }
    return meanBiases;
  }

  private void updateL2Weights(double[] meanL2Gradients, double meanL2bias, int currentBatchSize) {

    for (int i = 0; i < L2weights.length; i++) {

      if (i < L2weights.length - 1) {
        L2weights[i] = L2weights[i] - (this.learningRate * (meanL2Gradients[i] / currentBatchSize));
      }
      // update bias
      else {
        L2weights[i] = L2weights[i] - (this.learningRate * (meanL2bias / currentBatchSize));
      }
    }
  }
  
  private double[] getL2Gradient(double output, int iteration, double[] hiddena_iArray, double[] meanGradients) {
    
      for (int k = 0; k < numberHiddenUnits; k++) {
        meanGradients[k] += ((output - labels.get(iteration)) * output * (1 - output) * hiddena_iArray[k]);
      }
    return meanGradients;
  }
  
  private double getL2bias(double output, int iteration) {
    
    return ((output - labels.get(iteration)) * output * (1 - output));
  }

  private double totalCost(double[] output, int batchBegin, int batchEnd) {

    double cost = 0;
    for (int i = 0; i < batchEnd - batchBegin; i++) {

      cost += Math.pow((output[i] - labels.get(batchBegin+i)), 2);
    }
    return cost * 0.5;
  }

  public int getPrediction(ArrayList<Double> imageData, double[][] officialL1weights,
      double[] officialL1biases, double[] officialL2weights, double officialL2bias) {

    double[] hiddenz_iArray = new double[numberHiddenUnits];
    double[] hiddena_iArray = new double[numberHiddenUnits];

    // get hidden z_i's
    double sum = 0;
    for (int k = 0; k < numberHiddenUnits; k++) {
      sum = 0;
      for (int i = 0; i < imageData.size(); i++) {

        sum += imageData.get(i) * officialL1weights[i][k];
      }
      // add bias
      sum += officialL1biases[k];
      hiddenz_iArray[k] = sum;
    }

    // get hidden a_i's
    for (int i = 0; i < hiddenz_iArray.length; i++) {

      hiddena_iArray[i] = 1 / (1 + Math.pow(Math.E, hiddenz_iArray[i] * -1));
    }

    // get layer 2 z_i's
    double z_i = 0;
    for (int i = 0; i < hiddena_iArray.length; i++) {

      z_i += hiddena_iArray[i] * officialL2weights[i];
    }
    // add bias
    z_i += officialL2bias;

    // get layer 2 a_i
    double a_i = 1 / (1 + Math.pow(Math.E, z_i * -1));
    System.out.printf("%.2f", a_i);
    System.out.print(",");

    if (a_i >= 0.5) {
      return 1;
    } else {
      return 0;
    }
  }

  public int getNumberHiddenUnits() {
    return numberHiddenUnits;
  }
}
