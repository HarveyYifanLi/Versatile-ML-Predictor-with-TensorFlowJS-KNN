const { Configuration, OpenAIApi } = require("openai");
require("dotenv").config();

// initialize OpenAI API configuration
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
// create an openai instance and export it
const openai = new OpenAIApi(configuration);

module.exports = { openai }