package src;
//////////// FILE HEADER ///////////////
// Author: Yash Hindka (yhindka@wisc.edu)
// CS540, Young Wu
// Lecture 002, Summer 2020
// Project: P1 Logistic Regression/Neural Network on MNIST Dataset
// 6/7/2020 ////////////////////////////

import java.util.ArrayList;

public class LogisticRegression {

  private ArrayList<Double> weights;
  private double learningRate;
  private ArrayList<Integer> labels;
  private ArrayList<ArrayList<Double>> trainingData;
  private double epsilon;

  public LogisticRegression(ArrayList<ArrayList<Double>> trainingData, ArrayList<Integer> labels) {

    this.weights = new ArrayList<Double>();
    // initialize random weights between -1 and 1
    for (int i = 0; i < trainingData.get(0).size() - 1; i++) {
      
      weights.add(Math.random() * 2 - 1);
    }
    // add bias
    weights.add(Math.random() * 2 - 1);
    
    this.learningRate = 0.001;
    this.labels = labels;
    this.trainingData = trainingData;
    this.epsilon = 0.0001;
  }

  public ArrayList<Double> regression(ArrayList<ArrayList<Double>> trainingData) {

    
    double cost = 0;
    double prevCost = 1;
    int iteration = 1;
    
    ArrayList<Double> prevWeights = this.weights;
    // repeat until converges or 1000 iterations
    while (Math.abs(cost - prevCost) > this.epsilon && iteration < 1000) {
      // store previous iteration data
      prevWeights = weights;
      
      // get current iteration data
      ArrayList<Double> a_iArray = evaluateActivationFunction(trainingData);
      updateWeights(a_iArray, trainingData, iteration);
      prevCost = cost;
      cost = totalCost(a_iArray);
      iteration++;
    }
    
    // need data from previous iteration b/c that was minimum cost
    weights = prevWeights;
    
    return weights;
  }

  private ArrayList<Double> evaluateActivationFunction(ArrayList<ArrayList<Double>> trainingData) {

    
    ArrayList<Double> z_iArray = new ArrayList<>();
    ArrayList<Double> a_iArray = new ArrayList<>();
    
    // get z_i's
    for (int i = 0; i < labels.size(); i++) {
      double sum = 0;
      for (int j = 0; j < weights.size(); j++) {
        
        if (j < weights.size() - 1) {
          sum += weights.get(j) * trainingData.get(i).get(j);
        }
        
        // add bias
        else {
          sum += weights.get(j);
        }
      }
      z_iArray.add(sum);
    }
    
    // get a_i's
    for (int i = 0; i < z_iArray.size(); i++) {

      a_iArray.add(1 / (1 + Math.pow(Math.E, z_iArray.get(i) * -1)));
    }
    return a_iArray;
  }
  
  private void updateWeights(ArrayList<Double> a_iArray, ArrayList<ArrayList<Double>> trainingData, int iteration) {
    
    this.learningRate = this.learningRate / Math.sqrt(iteration);

    for (int i = 0; i < weights.size() - 1; i++) {
        
      if (i < weights.size() - 1) {
        weights.set(i, weights.get(i) - learningRate * getGradient(a_iArray, i, false));
      }
      else {
        weights.set(weights.size() - 1, (weights.get(weights.size() - 1)) - learningRate * getGradient(a_iArray, weights.size() - 1, true));
      }
    }
  }

  private double getGradient(ArrayList<Double> a_iArray, int column, boolean bias) {

    double sum = 0;
    if (!bias) {
      for (int i = 0; i < a_iArray.size(); i++) {
        
        sum += ((a_iArray.get(i) - this.labels.get(i)) * trainingData.get(i).get(column));
      }
    }
    else {
      for (int i = 0; i < a_iArray.size(); i++) {
        
        sum += (a_iArray.get(i) - this.labels.get(i));
      }
    }
    return sum;
  }

  private double totalCost(ArrayList<Double> a_iArray) {

    double cost = 0;
    for (int i = 0; i < labels.size(); i++) {

      if (labels.get(i) == 0 && (a_iArray.get(i) < 0.0001 || a_iArray.get(i) > 0.9999)) {
        cost += 100;
      } else if (a_iArray.get(i) == 0) {
        cost += Math.log10(1 - a_iArray.get(i));
      } else if (a_iArray.get(i) == 1) {
        cost += Math.log(a_iArray.get(i));
      } else {
        cost +=
          (labels.get(i) * Math.log10(a_iArray.get(i))) + ((1.0 - labels.get(i)) * Math.log10(1 - a_iArray.get(i)));
      }
    }
    return cost * -1;
  }
  
  public int getPrediction(ArrayList<Double> officialWeights, ArrayList<Double> imageData, ArrayList<Double> a_iArray) {
    
    // get z_i's
    double sum = 0;
    for (int i = 0; i < officialWeights.size(); i++) {
      
      if (i < officialWeights.size() - 1) {
        sum += officialWeights.get(i) * imageData.get(i);
      }
      // add bias
      else {
        sum += officialWeights.get(i);
      }
    }
    
    // get a_i
    double a_i = 1 / (1 + Math.pow(Math.E, sum * -1));
    a_iArray.add(a_i);
    if (a_i >= 0.5) {
      return 1;
    }
    else {
      return 0;
    }
  }



}
