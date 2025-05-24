const express = require('express');
const router = express.Router();

router.post('/', (req, res) => {
  const { destination, categories } = req.body;
  console.log('Received destination:', destination);
  console.log('Received categories:', categories);

  // Hardcoded places for Pune based on categories
  const hardcodedPlaces = {
    temple: [
      'Shreemant Dagdusheth Halwai Ganpati Temple',
      'Parvati Hill Temple',
      'Chaturshringi Temple',
      'Pataleshwar Cave Temple'
    ],
    cultural: [
      'Raja Dinkar Kelkar Museum',
      'Aga Khan Palace',
      'Shaniwar Wada',
      'Lal Mahal'
    ],
    cafe: [
      'Vohuman Cafe',
      'Cafe Goodluck',
      'Pagdandi Books Chai Cafe',
      'The Flour Works'
    ],
    park: [
      'Pune Okayama Friendship Garden',
      'Bund Garden',
      'Empress Botanical Garden',
      'Saras Baug'
    ]
  };

  if (destination !== 'Pune') {
    return res.json({ places: ['No places found for the given destination.'] });
  }

  const places = categories
    .flatMap(category => hardcodedPlaces[category] || [])
    .slice(0, 10); // Limit to 10 places

  res.json({ places: places.length ? places : ['No places found for the selected categories.'] });
});

module.exports = router;


