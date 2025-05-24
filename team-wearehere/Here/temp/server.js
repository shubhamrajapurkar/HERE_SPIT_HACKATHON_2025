const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const geminiProxy = require('./gemini-proxy');

const app = express();

// Enable CORS
app.use(cors());

// Parse JSON request bodies
app.use(bodyParser.json());

// Route for hardcoded places
app.use('/gemini-proxy', geminiProxy);

const PORT = 3001;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
