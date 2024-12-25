const { OpenAI } = require("openai");
require("dotenv").config();

// initialize OpenAI API configuration pojo
const configuration = {
  apiKey: process.env.OPENAI_API_KEY,
};
// create an openai instance and export it
const openai = new OpenAI(configuration);

/**
 * generate vector embeddings for a given string using OpenAI API.
 * @param {string} inputString - The string to be embedded.
 * @returns {Promise<number[]>} - The embedding vector.
 */
async function getEmbedding(inputString) {
    try {
      const response = await openai.embeddings.create({
        model: "text-embedding-ada-002",
        input: inputString,
      });

      return response.data[0].embedding; // Return the embedding vector, i.e. an array of numbers
    } catch (error) {
      console.error("Error generating embedding:", error.message);
      throw error;
    }
}

module.exports = { openai, getEmbedding }