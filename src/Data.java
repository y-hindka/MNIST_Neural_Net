package src;
//////////// FILE HEADER ///////////////
// Author: Yash Hindka (yhindka@wisc.edu)
// CS540, Young Wu
// Lecture 002, Summer 2020
// Project: P1 Logistic Regression/Neural Network on MNIST Dataset
// 6/7/2020 ////////////////////////////

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Data {

  public static void getFeatureMatrix(String fileName, List<Integer> setsToExtract,
      ArrayList<ArrayList<Double>> featureMatrix, ArrayList<Integer> Labels)
      throws FileNotFoundException {

    Scanner scn = new Scanner(new File(fileName));

    while (scn.hasNextLine()) {
      String line = scn.nextLine();
      boolean extract = false;
      // check if line should be read
      for (Integer check : setsToExtract) {
        if (check.equals(Integer.parseInt(line.substring(0, 1)))) {
          extract = true;
          break;
        }
      }
      if (extract) {
        // label 0 if 5, label 1 if 6
        if (line.substring(0, 1).equals("5")) {
          Labels.add(0);
        } else {
          Labels.add(1);
        }

        ArrayList<Double> toAdd = new ArrayList<Double>();
        line = line.substring(2);
        String[] lineArray = line.split(",");
        // convert strings to pixel intensity values between 0 and 1
        for (String pixel : lineArray) {
          toAdd.add(Double.valueOf(pixel) / 255.0);
        }
        featureMatrix.add(toAdd);
      }
    }

    scn.close();

  }

  public static ArrayList<ArrayList<Double>> getPixelList(String fileName)
      throws FileNotFoundException {

    Scanner scn = new Scanner(new File(fileName));
    ArrayList<ArrayList<Double>> pixelList = new ArrayList<>();
    pixelList.add(new ArrayList<Double>());

    int pixelCount = 0;
    int listIndex = 0;
    while (scn.hasNextLine()) {

      String[] lineArray = scn.nextLine().split(",");
      for (String pixel : lineArray) {
        pixelList.get(listIndex).add(Double.valueOf(pixel) / 255.0);
        pixelCount++;
      }
      if (pixelCount >= 784) {
        listIndex++;
        pixelList.add(new ArrayList<Double>());
        pixelCount = 0;
      }
    }
    
    // remove any lists with length 0
    if (pixelList.get(pixelList.size() - 1).size() == 0) {
      pixelList.remove(pixelList.size() - 1);
    }
    scn.close();

    return pixelList;
  }

}
