const fs = require('fs');
const _ = require('lodash');
const shuffleSeed = require('shuffle-seed');
const { getEmbedding } = require('./llms');
const { calculateEntropy} = require('./computation_helpers');

function extractColumns(data, columnNames) {
  // data is an array of arrays
  const headers = _.first(data);
  // get the indices of all the columns that we want to extract
  const indexes = _.map(columnNames, column => headers.indexOf(column));
  // for each row of data, just pull out the elements at specified indexes. https://www.geeksforgeeks.org/lodash-_-pullat-method/
  const extracted = _.map(data, row => _.pullAt(row, indexes));

  return extracted;
}

module.exports = async function loadCSV(
  filename,
  {
    dataColumns = [], // i.e. an array of header strings representing the features
    labelColumns = [], // i.e. an array of header strings representing the labels
    converters = {},
    shuffle = false,
    splitTest = false,
    useLlmEmbedding = false,
  }
) {
    let data = fs.readFileSync(filename, { encoding: 'utf-8' }); // returns a big string

    data = _.map(data.split('\n'), d => d.split(',')); // data is now an array of arrays

    data = _.dropRightWhile(data, val => _.isEqual(val, ['']));

    const headers = _.first(data);
    console.log('headers', headers); // e.g. headers: ['"Renewal ARR"', '"Monthly Amount"', '"Age"', '"Tier at close"', '"Account Count"', '"Signup Date"']

    if (!useLlmEmbedding) {
      // parse each column of each row directly without using LLM vector embedding
      data = _.map(data, (row, index) => {
        // i.e. For each row:
        if (index === 0) {
          // skip while on the headers row
          return row;
        }
  
        const convertedRow =  _.map(row, (element, index) => {
          // i.e. For each column on this row:
          // console.log('element', element); // i.e. element represents the data on each column on a certain row
          // console.log('converters[headers[index]]', converters[headers[index]]); // i.e. converters[headers[index]] returns the customized function for that header string that was sent as input when calling loadCSV()
          // if a custom converter function exists, use it to parse the value
          if (converters[headers[index]]) {
            const converted = converters[headers[index]](element);
  
            return _.isNaN(converted) ? 0 : converted;
          }
          // check if current column's value is already a number "in essence"
          const columnOFNumberValue = !!(typeof element === 'string' && parseFloat(element));
  
          let result;
          // if column is already a number "in essence", parse it directly
          // else return 0
          if (columnOFNumberValue) {
            result = parseFloat(element.replaceAll('"', '').replaceAll('\'', ''));
          } else {
            result = 0;
          }
  
          return _.isNaN(result) ? 0 : result;
        });

        return convertedRow;
      });
    } else { // use LLM vector embedding for a column that's non-numeric
      //resolve outer array of promises each of which is from processing each row's data
      data = await Promise.all(
        _.map(data, async (row, index) => {
          // i.e. For each row:
          if (index === 0) {
            // skip while on the headers row
            return row;
          }
          // resolve inner array of promises each of which is from processing each column's data
          const convertedRow = await Promise.all(
            _.map(row, async (element, index) => {
              // i.e. For each column on this row:
              // if a custom converter function exists, use it to parse the value
              if (converters[headers[index]]) {
                const converted = converters[headers[index]](element);
      
                return _.isNaN(converted) ? 0 : converted;
              }
              // check if current column's value is already a number "in essence"
              const columnOFNumberValue = !!(typeof element === 'string' && parseFloat(element));
      
              let result;
              // if column is already a number "in essence", parse it directly
              // else use LLM vector embedding for it, then calculate and return its entropy as a number
              if (columnOFNumberValue) {
                result = parseFloat(element.replaceAll('"', '').replaceAll('\'', ''));
              } else {
                console.log('------------------- Vector Embedding Process Initiated -----------------------------');
                const embedding = await getEmbedding(element);
                // console.log('embedding', embedding);
                const entropyOfEmbedding = calculateEntropy(embedding);
                // console.log('entropyOfEmbedding', entropyOfEmbedding);
                result = entropyOfEmbedding;
              }
      
              return _.isNaN(result) ? 0 : result;
            })
          );

          return convertedRow;
        })
      );
    }

    let labels = extractColumns(data, labelColumns); // i.e. the labels, nested 2D array, of the same length as the features
    data = extractColumns(data, dataColumns); // i.e. the features, nested 2D array, of the same length as the labels

    data.shift(); // remove the first row, which are headers
    labels.shift(); // remove the first row, which are headers

    // Randomly shuffle the datasets
    if (shuffle) {
      const shufflePhase = 'E=MC^2';
      // Shuffle the features and labels in the EXACT SAME way (by using the same shuffle phase) so as not to loose its relative order connections
      data = shuffleSeed.shuffle(data, shufflePhase);
      labels = shuffleSeed.shuffle(labels, shufflePhase);
    }

    // Split datasets into training sets and test sets
    if (splitTest) {
      const testSetSize = _.isNumber(splitTest) ? splitTest : Math.floor(data.length / 5);
      console.log('testSetSize', testSetSize);

      const startIndex = 0;
      const endIndex = data.length;

      return {
        features: data.slice(startIndex, endIndex - testSetSize),
        labels: labels.slice(startIndex, endIndex - testSetSize),
        testFeatures: data.slice(endIndex - testSetSize, endIndex),
        testLabels: labels.slice(endIndex - testSetSize, endIndex),
      };
    }
    else {
      return { features: data, labels };
    }
};
