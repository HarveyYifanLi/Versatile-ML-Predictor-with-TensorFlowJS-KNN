require('@tensorflow/tfjs-node'); // use CPU rather than GPU
const tf = require('@tensorflow/tfjs');

/** The calculation of the mean and variance tensors for the features dataset
* params featuresTensor: a 2D tensor
* params operationAxis: a number i.e. 0 or 1
* returns { mean: a 1D tensor, variance: a 1D tensor }
*/
function getMeanAndVarianceTensors(featuresTensor, operationAxis) {
    const { mean, variance } = tf.moments(featuresTensor, operationAxis); // tf.moments returns the mean and variance of an input tensor. Similar to sum(1, true), need to calculate the mean and variance for all the rows(i.e. along 0-axis)
    console.log('The mean tensor:');
    mean.print();

    console.log('The standard deviation tensor:');
    variance.pow(0.5).print();

    if (mean && variance) {
     return { mean, variance }
    }
    return {}
}

/** The implementation of K-Nearest-Neighbour algorithm
* params features: a 2D tensor
* params labels: a 2D tensor
* params predictionPoint: a 1D tensor
* params k: integer
* params mean: a 1D tensor
* params variance: a 1D tensor
* returns predictedResult: a number
*/
function trainWithKNN(features, labels, featureTensorForPrediction, k, mean, variance) {
 // Standardized Value = (value - mean)/StandardDeviation 
 // (and variance = SD^2)
 const scaledFeatureForPrediction = featureTensorForPrediction.sub(mean).div(variance.pow(0.5));

 // Same as above, use numerical standardization to normalized the features dataset
 const scaledFeatures = 
     features
     .sub(mean)
     .div(variance.pow(0.5));
 // Implement KNN with tensor calculation
 const predictedResult = 
     scaledFeatures
     .sub(scaledFeatureForPrediction)
     .pow(2)
     .sum(1) // sum along 1-axis, i.e. returns a 1D tensor of shape [n]
     .pow(0.5)
     .expandDims(1) // to change the 1D tensor of shape [n] into [n, 1]
     .concat(labels, 1) // concat with labels tensor (of shape [n, 1]) along 1-axis
     .unstack()// to transform this 2D tensor an array of 1D tensors
     .sort((a, b) => a.get(0) > b.get(0) ? 1 : -1) // a and b is a Tensor, thus need to call .get(0) to obtain the first value of current 1D tensor(e.g. [0.5, 200]) 
     .slice(0, k) // take the first k records (i.e. k of those 1D tensors)
     .reduce((acc, currentDataPair) => acc + currentDataPair.get(1), 0) / k; // average the 'label' (i.e. currentDataPair.get(1)) values of those top k records

 return predictedResult;
}

/** From Information Theory: Calculate the Entropy of a Random Variable that is an array of numbers
* params embedding: an array of numbers
* returns total: the Entropy of this embedding Random Variable accordingly to Information Theory
*/
function calculateEntropy(embedding) {
    const total = embedding.reduce((sum, value) => sum + Math.abs(value), 0);
    return -embedding
      .map((value) => Math.abs(value) / total)
      .reduce((entropy, prob) => (prob > 0 ? entropy + prob * Math.log2(prob) : entropy), 0);
}

module.exports = { getMeanAndVarianceTensors, trainWithKNN, calculateEntropy };
