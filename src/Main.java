package src;
//////////// FILE HEADER ///////////////
// Author: Yash Hindka (yhindka@wisc.edu)
// CS540, Young Wu
// Lecture 002, Summer 2020
// Project: P1 Logistic Regression/Neural Network on MNIST Dataset
// 6/7/2020 ////////////////////////////

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;

public class Main {

  public static void main(String[] args) {

    ArrayList<Integer> setsToExtract = new ArrayList<>(Arrays.asList(5, 6)); // NN will try to distinguish between labels 5 and 6
    ArrayList<ArrayList<Double>> trainingData = new ArrayList<>();
    ArrayList<Integer> labels = new ArrayList<>(); // labels are the number the pixels represent (0 - 9)
    ArrayList<Double> weights = new ArrayList<>();

    try {
      Data.getFeatureMatrix("data/mnist_train.csv", setsToExtract, trainingData, labels);
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }

    // get minimum weights by performing logistic regression
    
     LogisticRegression lr = new LogisticRegression(trainingData, labels); 
     weights = lr.regression(trainingData);
      
     // print weights 
     for (Double weight : weights) { 
       System.out.printf("%.4f", weight);
       System.out.print(","); 
     }
      
     System.out.println();
     

    // get test data
    ArrayList<ArrayList<Double>> testData = new ArrayList<>();
    try {
      testData = Data.getPixelList("data/test.txt");
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }

    
     // get predictions 
    ArrayList<Integer> predictions = new ArrayList<>(); 
    ArrayList<Double> activations = new ArrayList<>(); 
    for (int i = 0; i < testData.size(); i++) {
      
     if (testData.get(i).size() == 784) { 
       predictions.add(lr.getPrediction(weights,testData.get(i), activations)); 
       } 
    }
      
     // print activations 
    /*for (Double activation : activations) { 
      System.out.printf("%.2f", activation); 
      System.out.print(","); 
    }*/
      
    // print predictions 
    System.out.println("Logistic Regression Predictions");
    for (Integer prediction : predictions) {
     System.out.print(prediction + ","); 
    }

    System.out.println("\nNEURAL NETWORK");

    /// Neural Network ///  

    NeuralNetwork n = new NeuralNetwork(trainingData, labels);
    double[][] allWeights = n.backpropagation();
    double[][] officialL1weights = new double[allWeights.length - 2][n.getNumberHiddenUnits()];
    double[] officialL1biases = new double[n.getNumberHiddenUnits()];
    double[] officialL2weights = new double[n.getNumberHiddenUnits() + 1];

    // fill L1 and L2 weights and biases
    for (int r = 0; r < allWeights.length; r++) {
      for (int c = 0; c < allWeights[r].length; c++) {
        if (r < allWeights.length - 2 && c < allWeights[r].length - 1) {
          officialL1weights[r][c] = allWeights[r][c];
        } else if (r == allWeights.length - 2 && c < allWeights[r].length - 1) {
          officialL1biases[c] = allWeights[r][c];
        } else if (r == allWeights.length - 1){
          officialL2weights[c] = allWeights[r][c];
        }
      }
    }
    
    PrintWriter pw = null;
    try {
      pw = new PrintWriter(new File("results/NNData.txt"));
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
    
    // print weights/biases
    pw.write("Layer 1 weights:");
    for (int r = 0; r < officialL1weights.length; r++) {
      for (int c = 0; c < officialL1weights[r].length; c++) {
        pw.printf("%.4f", officialL1weights[r][c]);
        pw.print(",");
      }
      pw.println();
      pw.flush();
    }
    pw.println("\nLayer 1 biases");
    for (int i = 0; i < officialL1biases.length; i++) {
      pw.printf("%.4f", officialL1biases[i]);
      pw.print(",");
    }
    pw.flush();
    pw.println("\nLayer 2 weights");
    for (int i = 0; i < officialL2weights.length; i++) {
      pw.printf("%.4f", officialL2weights[i]);
      pw.print(",");
    }
    pw.flush();
    
    int[] neuralPredictions = new int[testData.size()];

    System.out.println("\nActivations");
    for (int i = 0; i < testData.size(); i++) {
      if (testData.get(i).size() == 784) {
        neuralPredictions[i] = n.getPrediction(testData.get(i), officialL1weights,
            officialL1biases, officialL2weights, officialL2weights[officialL2weights.length - 1]);
      }
    }

    pw.println("Neural Net Predictions:");
    for (int pred : neuralPredictions) {
      pw.print(pred + ",");
    }
    pw.flush();
    pw.close();

  }

}
