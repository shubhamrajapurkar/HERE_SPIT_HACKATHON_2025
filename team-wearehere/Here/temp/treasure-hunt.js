mapboxgl.accessToken = 'your-mapbox-access-token'; // Replace with your Mapbox token

// Initialize the map
const map = new mapboxgl.Map({
  container: 'map',
  style: 'mapbox://styles/mapbox/streets-v11',
  center: [73.8567, 18.5204], // Centered on Pune
  zoom: 12
});

// Hardcoded treasures for local businesses with hints
const treasures = [
  {
    name: 'Cafe Goodluck',
    coords: [73.8415, 18.5167],
    reward: '10% off on your next order!',
    hints: ['A legendary cafe in Pune', 'Famous for bun maska and Irani chai', 'Located near Deccan Gymkhana']
  },
  {
    name: 'Pagdandi Books Chai Cafe',
    coords: [73.8077, 18.5094],
    reward: 'Free chai with any book purchase!',
    hints: ['A cozy cafe for book lovers', 'Located in Baner', 'Known for its chai and peaceful ambiance']
  },
  {
    name: 'The Flour Works',
    coords: [73.9025, 18.5545],
    reward: 'Buy 1 Get 1 Free on desserts!',
    hints: ['A European-style cafe', 'Located in Kalyani Nagar', 'Famous for its breakfast and desserts']
  },
  {
    name: 'Bund Garden',
    coords: [73.8768, 18.5393],
    reward: 'Free boat ride for 2!',
    hints: ['A serene garden in Pune', 'Located near Yerawada', 'Known for its boating activities']
  }
];

let currentTreasureIndex = 0;

// Display the current clue and hints
function showClue() {
  const treasureClue = document.getElementById('treasure-clue');
  const treasureHints = document.getElementById('treasure-hints');
  if (currentTreasureIndex < treasures.length) {
    const treasure = treasures[currentTreasureIndex];
    treasureClue.innerHTML = `<span class="font-bold text-indigo-700">Find:</span> ${treasure.name}`;
    treasureHints.innerHTML = `
      <h3 class="text-lg font-bold text-indigo-700 mb-2">Hints:</h3>
      <ul class="list-disc pl-5 text-gray-700">
        ${treasure.hints.map(hint => `<li>${hint}</li>`).join('')}
      </ul>`;
  } else {
    treasureClue.innerHTML = '<span class="font-bold text-green-700">Congratulations!</span> You have found all the treasures!';
    treasureHints.innerHTML = '';
  }
}

// Check if the user clicked near the treasure
function checkProximity(clickedCoords) {
  if (currentTreasureIndex >= treasures.length) return; // All treasures found

  const treasure = treasures[currentTreasureIndex];
  const distance = Math.sqrt(
    Math.pow(clickedCoords[0] - treasure.coords[0], 2) +
    Math.pow(clickedCoords[1] - treasure.coords[1], 2)
  );

  // If the user is close enough, mark the treasure as found
  if (distance < 0.01) { // Adjust proximity threshold as needed
    promotePlace(treasure);
    currentTreasureIndex++;
    showClue();
  } else {
    alert('Not quite! Try clicking closer to the location.');
  }
}

// Promote the place when found
function promotePlace(treasure) {
  const promotion = document.getElementById('promotion');
  promotion.innerHTML = `
    <h3 class="text-lg font-bold text-green-700 mb-2">Congratulations!</h3>
    <p>You found <strong>${treasure.name}</strong>!</p>
    <p class="text-gray-700">${treasure.reward}</p>
  `;
  promotion.style.display = 'block';

  // Hide promotion after 5 seconds
  setTimeout(() => {
    promotion.style.display = 'none';
  }, 5000);
}

// Add a click event to the map
map.on('click', (e) => {
  const clickedCoords = [e.lngLat.lng, e.lngLat.lat];
  checkProximity(clickedCoords);
});

// Add functionality to the "Back" button
document.getElementById('back-btn').onclick = function () {
  window.location.href = 'http://127.0.0.1:5500/mapbox.html?src=Kalyan&dest=Pune'; // Redirect to the main map page
};

// Display the first clue
showClue();
