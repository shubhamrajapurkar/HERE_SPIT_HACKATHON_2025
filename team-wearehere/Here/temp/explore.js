// Wait for mapbox.js to finish initializing the map and directions
window.addEventListener('DOMContentLoaded', () => {
  let map, directions, dest;

  function getDestLngLat() {
    // Try to get destination coordinates from directions input
    if (window.prefillDirections && window.prefillDirections.dest) {
      return window.prefillDirections.dest;
    }
    return null;
  }

  // Show the Explore Destination button after route is loaded
  function showExploreDestBtn() {
    document.getElementById('explore-dest-btn').style.display = 'block';
  }

  // Wait for mapbox.js to set up the map and directions
  function tryInit() {
    map = window.mapboxMap;
    directions = window.mapboxDirections;
    dest = getDestLngLat();
    if (directions && dest) {
      showExploreDestBtn();
    } else {
      setTimeout(tryInit, 400);
    }
  }
  tryInit();

  // Explore Destination button logic
  document.getElementById('explore-dest-btn').onclick = function() {
    if (window.prefillDirections && window.prefillDirections.dest && map && directions) {
      directions.on('route', (e) => {
        if (e.route && e.route[0] && e.route[0].legs && e.route[0].legs[0].steps.length > 0) {
          const lastStep = e.route[0].legs[0].steps[e.route[0].legs[0].steps.length - 1];
          const coords = lastStep.maneuver.location;
          map.flyTo({ center: coords, zoom: 15 });
        }
      });
      directions.setDestination(window.prefillDirections.dest);
    }
    // Show "Explore Nearby Places" button
    document.getElementById('explore-nearby-btn').style.display = 'block';
    this.style.display = 'none';
  };

  // Explore Nearby Places button
  document.getElementById('explore-nearby-btn').onclick = function () {
    const exploreModal = document.getElementById('explore-modal');
    exploreModal.style.display = 'flex';

    // Ensure modal footer exists
    let modalFooter = document.getElementById('explore-modal-footer');
    if (!modalFooter) {
      modalFooter = document.createElement('div');
      modalFooter.id = 'explore-modal-footer';
      modalFooter.className = 'mt-4 text-center';
      exploreModal.appendChild(modalFooter);
    }

    // Add "Video Call with Guide" button dynamically
    if (!document.getElementById('video-call-btn')) {
      const videoCallBtn = document.createElement('button');
      videoCallBtn.id = 'video-call-btn';
      videoCallBtn.textContent = 'Video Call with Guide';
      videoCallBtn.className = 'bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg mt-4';
      videoCallBtn.onclick = function () {
        window.location.href = 'videocall.html'; // Redirect to video call page
      };
      modalFooter.appendChild(videoCallBtn);
    }
  };

  // Modal close
  document.getElementById('close-modal').onclick = function() {
    document.getElementById('explore-modal').style.display = 'none';
  };

  // Hardcoded places for Pune with coordinates and detailed facts
  const hardcodedPlaces = {
    temple: [
      {
        name: 'Shreemant Dagdusheth Halwai Ganpati Temple',
        coords: [73.8567, 18.5196],
        model: 'models/dagdusheth.glb',
        fact: 'A famous Ganesh temple built over 125 years ago. It is a symbol of faith and devotion in Pune. The temple is known for its grand Ganesh idol adorned with gold and precious ornaments. It attracts thousands of devotees during the Ganesh festival. The temple is located in the heart of Pune near Shaniwar Wada.'
      },
      {
        name: 'Parvati Hill Temple',
        coords: [73.8521, 18.5010],
        model: 'models/parvati.glb',
        fact: 'Located on a hill, this temple offers a panoramic view of Pune city. It is one of the oldest heritage structures in Pune, built during the Peshwa dynasty. The temple is dedicated to Lord Shiva and Goddess Parvati. Visitors often climb the 103 steps to reach the temple and enjoy the serene atmosphere.'
      },
      {
        name: 'Chaturshringi Temple',
        coords: [73.8291, 18.5382],
        model: 'models/chaturshringi.glb',
        fact: 'Dedicated to Goddess Chaturshringi, this temple symbolizes power and faith. It is located on a hill and requires a climb of 100 steps to reach. The temple is a popular destination during the Navratri festival. It is believed to have been built during the reign of the Marathas.'
      },
      {
        name: 'Pataleshwar Cave Temple',
        coords: [73.8478, 18.5204],
        model: 'models/pataleshwar.glb',
        fact: 'An 8th-century rock-cut temple dedicated to Lord Shiva. It is carved out of a single basalt rock and features intricate sculptures. The temple is a fine example of ancient Indian rock-cut architecture. It is located in the heart of Pune and offers a peaceful retreat from the bustling city.'
      }
    ],
    cultural: [
      {
        name: 'Raja Dinkar Kelkar Museum',
        coords: [73.8553, 18.5104],
        model: 'models/kelkar_museum.glb',
        fact: 'This museum houses a vast collection of artifacts from across India. It was established by Dr. Dinkar Kelkar in memory of his son. The museum features over 20,000 objects, including musical instruments, paintings, and sculptures. It provides a glimpse into India’s rich cultural heritage.'
      },
      {
        name: 'Aga Khan Palace',
        coords: [73.9020, 18.5523],
        model: 'models/aga_khan.glb',
        fact: 'A historical landmark where Mahatma Gandhi was imprisoned during the freedom struggle. The palace was built in 1892 by Sultan Aga Khan III. It is known for its Italian arches and sprawling gardens. The palace also houses a museum showcasing artifacts related to Gandhi’s life.'
      },
      {
        name: 'Shaniwar Wada',
        coords: [73.8567, 18.5194],
        model: 'models/shaniwar_wada.glb',
        fact: 'A 17th-century fortification and a symbol of Maratha pride. It was the seat of the Peshwas of the Maratha Empire. The fort is known for its majestic architecture and historical significance. It is a popular tourist destination and hosts a light and sound show in the evenings.'
      },
      {
        name: 'Lal Mahal',
        coords: [73.8570, 18.5186],
        model: 'models/lal_mahal.glb',
        fact: 'The childhood home of Chhatrapati Shivaji Maharaj. It was built by Shahaji Raje Bhosale for his wife Jijabai and son Shivaji. The palace has been reconstructed and features exhibits related to Shivaji’s life. It is located near Shaniwar Wada in Pune.'
      }
    ],
    cafe: [
      {
        name: 'Vohuman Cafe',
        coords: [73.8768, 18.5362],
        fact: 'Known for its iconic Irani chai and cheese omelets. This cafe is a favorite breakfast spot for locals and tourists alike. It has a simple and old-school ambiance. The cafe is located near Pune Railway Station and has been serving customers for decades.'
      },
      {
        name: 'Cafe Goodluck',
        coords: [73.8415, 18.5167],
        fact: 'One of Pune’s oldest cafes, serving delicious bun maska and Irani chai. It is a popular hangout spot for students and families. The cafe has a nostalgic charm and is located near Deccan Gymkhana. It is known for its quick service and affordable prices.'
      },
      {
        name: 'Pagdandi Books Chai Cafe',
        coords: [73.8077, 18.5094],
        fact: 'A cozy cafe for book lovers and chai enthusiasts. It offers a peaceful ambiance with a collection of books to read. The cafe is located in Baner and is a great place to relax and unwind. It also hosts events like book readings and poetry sessions.'
      },
      {
        name: 'The Flour Works',
        coords: [73.9025, 18.5545],
        fact: 'Famous for its European-style breakfast and desserts. This cafe is located in Kalyani Nagar and has a charming outdoor seating area. It is known for its freshly baked bread and pastries. The cafe is a favorite spot for brunch and coffee lovers.'
      }
    ],
    park: [
      {
        name: 'Pune Okayama Friendship Garden',
        coords: [73.8077, 18.5074],
        fact: 'A Japanese-style garden symbolizing Indo-Japanese friendship. It features beautiful landscapes, ponds, and walking paths. The garden is a peaceful retreat in the bustling city. It is located in Sinhagad Road and is a popular spot for picnics and photography.'
      },
      {
        name: 'Bund Garden',
        coords: [73.8768, 18.5393],
        fact: 'A serene spot for picnics and boating. The garden is located near Yerawada and is surrounded by lush greenery. It is a great place to relax and enjoy nature. The garden also features a jogging track and a children’s play area.'
      },
      {
        name: 'Empress Botanical Garden',
        coords: [73.8768, 18.5018],
        fact: 'A lush garden established during the British era. It is home to a variety of plants and trees. The garden also hosts flower shows and exhibitions. It is located near Pune Race Course and is a favorite spot for nature lovers.'
      },
      {
        name: 'Saras Baug',
        coords: [73.8530, 18.5018],
        fact: 'A historic garden with a temple dedicated to Lord Ganesh. It is a popular spot for evening walks and family outings. The garden is located near Swargate and features a large pond. It is a peaceful place to relax and enjoy the greenery.'
      }
    ]
  };

  // Hardcoded treasures for local businesses
  const treasures = [
    { name: 'Cafe Goodluck', coords: [73.8415, 18.5167], reward: '10% off on your next order!' },
    { name: 'Pagdandi Books Chai Cafe', coords: [73.8077, 18.5094], reward: 'Free chai with any book purchase!' },
    { name: 'The Flour Works', coords: [73.9025, 18.5545], reward: 'Buy 1 Get 1 Free on desserts!' },
    { name: 'Bund Garden', coords: [73.8768, 18.5393], reward: 'Free boat ride for 2!' }
  ];

  // Function to start the treasure hunt
  function startTreasureHunt() {
    const treasureClue = document.getElementById('treasure-clue');
    let currentTreasureIndex = 0;

    // Display the first clue
    function showClue() {
      if (currentTreasureIndex < treasures.length) {
        const treasure = treasures[currentTreasureIndex];
        treasureClue.textContent = `Clue: Find the treasure at ${treasure.name}`;
      } else {
        treasureClue.textContent = 'Congratulations! You have found all the treasures!';
      }
    }

    // Check if the user is near the treasure
    function checkProximity(userCoords) {
      if (currentTreasureIndex >= treasures.length) return;

      const treasure = treasures[currentTreasureIndex];
      const distance = Math.sqrt(
        Math.pow(userCoords[0] - treasure.coords[0], 2) +
        Math.pow(userCoords[1] - treasure.coords[1], 2)
      );

      // If the user is close enough, mark the treasure as found
      if (distance < 0.01) { // Adjust proximity threshold as needed
        alert(`You found the treasure at ${treasure.name}! Reward: ${treasure.reward}`);
        currentTreasureIndex++;
        showClue();
      }
    }

    // Simulate user movement (replace with actual GPS tracking)
    navigator.geolocation.watchPosition((position) => {
      const userCoords = [position.coords.longitude, position.coords.latitude];
      checkProximity(userCoords);
    });

    showClue();
  }

  // Add a button to start the treasure hunt
  const treasureHuntBtn = document.createElement('button');
  treasureHuntBtn.id = 'treasure-hunt-btn';
  treasureHuntBtn.textContent = 'Start Treasure Hunt';
  treasureHuntBtn.className = 'bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-lg mt-4';

  // Ensure the button is appended to a visible container
  let container = document.getElementById('explore-modal-footer');
  if (!container) {
    console.warn('Explore modal footer not found. Appending treasure hunt button to the explore modal.');
    container = document.getElementById('explore-modal');
  }
  if (!container) {
    console.warn('Explore modal not found. Appending treasure hunt button to the body.');
    container = document.body;
  }
  container.appendChild(treasureHuntBtn);

  // Redirect to the treasure hunt page
  treasureHuntBtn.onclick = function () {
    window.location.href = 'treasure-hunt.html';
  };

  // Add a clue display area
  const treasureClue = document.createElement('div');
  treasureClue.id = 'treasure-clue';
  treasureClue.className = 'mt-4 text-lg font-semibold text-indigo-700';
  document.body.appendChild(treasureClue);

  // Explore form submit
  document.getElementById('explore-form').onsubmit = function (e) {
    e.preventDefault();
    const checked = Array.from(document.querySelectorAll('input[name="category"]:checked')).map(cb => cb.value);
    if (!checked.length) {
      console.error('No categories selected.'); // Log if no categories are selected
      return;
    }
    document.getElementById('explore-modal').style.display = 'none';

    // Get destination (hardcoded to Pune for now)
    const dest = 'Pune';
    console.log('Destination:', dest); // Log the destination for debugging

    // Get places based on selected categories
    const places = checked
      .flatMap(category => hardcodedPlaces[category] || [])
      .slice(0, 10); // Limit to 10 places

    // Display places with facts
    const placesList = document.getElementById('places-list');
    if (places.length > 0) {
      placesList.innerHTML = `
        <h3 class="font-bold mb-2">Recommended Places</h3>
        <ul class="list-disc pl-5">
          ${places.map(place => `<li class="mb-1">${place.name}</li>`).join('')}
        </ul>`;
    } else {
      console.warn('No places found for the selected categories.'); // Log warning if no places are found
      placesList.innerHTML = '<div class="text-red-600">No places found for the selected categories.</div>';
    }
    placesList.style.display = 'block';

    // Hide the list after 5 seconds
    setTimeout(() => {
      placesList.style.display = 'none';
    }, 5000);

    // Pin places on the map
    pinPlacesOnMap(places);
  };

  // Initialize itinerary in localStorage
  if (!localStorage.getItem('itinerary')) {
    localStorage.setItem('itinerary', JSON.stringify([]));
  }

  // Function to pin places on the map
  function pinPlacesOnMap(places) {
    if (!map) {
      console.error('Map is not initialized.');
      return;
    }

    // Clear existing markers
    if (window.placeMarkers) {
      window.placeMarkers.forEach(marker => marker.remove());
    }
    window.placeMarkers = [];

    // Add markers for each place
    places.forEach(place => {
      const marker = new mapboxgl.Marker()
        .setLngLat(place.coords) // Use actual coordinates
        .addTo(map);

      // Add a popup with the place name, details, and "Add to Itinerary" button
      const popupContent = `
        <div class="p-2">
          <h3 class="text-lg font-bold text-indigo-700">${place.name}</h3>
          <p class="text-sm text-gray-600">${place.fact || 'No details available.'}</p>
          <button id="add-itinerary-${place.name}" class="mt-2 bg-green-600 hover:bg-green-700 text-white font-semibold py-1 px-2 rounded-lg">
            Add to Itinerary
          </button>
          <button id="open-3d-${place.name}" class="mt-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-1 px-2 rounded-lg">
            Open 3D Map
          </button>
        </div>
      `;
      const popup = new mapboxgl.Popup().setHTML(popupContent);
      marker.setPopup(popup);

      // Add click event for "Add to Itinerary" button
      marker.getElement().addEventListener('click', () => {
        setTimeout(() => {
          const addItineraryButton = document.getElementById(`add-itinerary-${place.name}`);
          if (addItineraryButton) {
            addItineraryButton.onclick = () => {
              const itinerary = JSON.parse(localStorage.getItem('itinerary'));
              if (!itinerary.some(p => p.name === place.name)) {
                itinerary.push(place);
                localStorage.setItem('itinerary', JSON.stringify(itinerary));
                alert(`${place.name} has been added to your itinerary.`);
              } else {
                alert(`${place.name} is already in your itinerary.`);
              }
            };
          }

          const open3DButton = document.getElementById(`open-3d-${place.name}`);
          if (open3DButton) {
            open3DButton.onclick = () => {
              const confirm3D = confirm(`Do you want to open the 3D map for ${place.name}?`);
              if (confirm3D) {
                const sketchfabUrl = 'https://sketchfab.com/models/d0df0ef8afc44c7b98fee535498a175e/embed';
                const modal = document.createElement('div');
                modal.id = 'sketchfab-modal';
                modal.style.position = 'fixed';
                modal.style.top = '0';
                modal.style.left = '0';
                modal.style.width = '100%';
                modal.style.height = '100%';
                modal.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
                modal.style.display = 'flex';
                modal.style.justifyContent = 'center';
                modal.style.alignItems = 'center';
                modal.style.zIndex = '1000';

                const iframe = document.createElement('iframe');
                iframe.src = sketchfabUrl;
                iframe.style.width = '80%';
                iframe.style.height = '80%';
                iframe.style.border = 'none';
                iframe.allow = 'autoplay; fullscreen; xr-spatial-tracking';

                const closeModal = document.createElement('button');
                closeModal.textContent = 'Close';
                closeModal.style.position = 'absolute';
                closeModal.style.top = '10px';
                closeModal.style.right = '10px';
                closeModal.style.padding = '10px';
                closeModal.style.backgroundColor = '#fff';
                closeModal.style.border = 'none';
                closeModal.style.cursor = 'pointer';

                closeModal.onclick = function () {
                  document.body.removeChild(modal);
                };

                modal.appendChild(iframe);
                modal.appendChild(closeModal);
                document.body.appendChild(modal);
              }
            };
          }
        }, 0);
      });

      window.placeMarkers.push(marker);
    });
  }

  // Move "Plan Itinerary" and "View Pune 3D Model" buttons above the map
  function addControlButtons() {
    const controlContainer = document.createElement('div');
    controlContainer.id = 'control-container';
    controlContainer.style.position = 'absolute';
    controlContainer.style.top = '10px';
    controlContainer.style.left = '10px';
    controlContainer.style.zIndex = '1000';

    // Add "Plan Itinerary" button
    const planItineraryBtn = document.createElement('button');
    planItineraryBtn.id = 'plan-itinerary-btn';
    planItineraryBtn.textContent = 'Plan Itinerary';
    planItineraryBtn.className = 'bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-4 rounded-lg shadow-lg mr-2';

    // Add "View Pune 3D Model" button
    const sketchfabBtn = document.createElement('button');
    sketchfabBtn.id = 'view-sketchfab-btn';
    sketchfabBtn.textContent = 'View Pune 3D Model';
    sketchfabBtn.className = 'bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg shadow-lg';

    // Append buttons to the container
    controlContainer.appendChild(planItineraryBtn);
    controlContainer.appendChild(sketchfabBtn);

    // Append the container to the body
    document.body.appendChild(controlContainer);

    // Add functionality to the buttons
    addPlanItineraryLogic(planItineraryBtn);
    addSketchfabLogic(sketchfabBtn);
  }

  // Function to add "Plan Itinerary" and "View 3D Model" buttons at the bottom
  function addBottomControlButtons() {
    const controlContainer = document.getElementById('bottom-control-container') || document.createElement('div');
    controlContainer.id = 'bottom-control-container';
    controlContainer.style.position = 'fixed';
    controlContainer.style.bottom = '20px';
    controlContainer.style.left = '50%';
    controlContainer.style.transform = 'translateX(-50%)';
    controlContainer.style.zIndex = '1000';
    controlContainer.style.display = 'flex';
    controlContainer.style.gap = '10px';

    // Remove all children to avoid duplicates
    while (controlContainer.firstChild) controlContainer.removeChild(controlContainer.firstChild);

    // "Plan Itinerary" button
    const planItineraryBtn = document.createElement('button');
    planItineraryBtn.id = 'plan-itinerary-btn';
    planItineraryBtn.textContent = 'Plan Itinerary';
    planItineraryBtn.className = 'bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-4 rounded-lg shadow-lg';

    // "View 3D Model" button
    const sketchfabBtn = document.createElement('button');
    sketchfabBtn.id = 'view-sketchfab-btn';
    sketchfabBtn.textContent = 'View 3D Model';
    sketchfabBtn.className = 'bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg shadow-lg';

    // "Automated Booking" button
    const automatedBookingBtn = document.createElement('button');
    automatedBookingBtn.id = 'automated-booking-btn';
    automatedBookingBtn.textContent = 'Automated Booking';
    automatedBookingBtn.className = 'bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded-lg shadow-lg';

    controlContainer.appendChild(planItineraryBtn);
    controlContainer.appendChild(sketchfabBtn);
    controlContainer.appendChild(automatedBookingBtn);

    // Only append if not already in DOM
    if (!document.getElementById('bottom-control-container')) {
        document.body.appendChild(controlContainer);
    }

    addPlanItineraryLogic(planItineraryBtn);
    addSketchfabLogic(sketchfabBtn);

    // Fix: Redirect to call.html and let user click the automate button there
    automatedBookingBtn.onclick = function() {
        window.location.href = 'call.html';
    };
  }

  // Add logic for "Plan Itinerary" button
  function addPlanItineraryLogic(planItineraryBtn) {
    planItineraryBtn.onclick = async () => {
      // Check if localStorage is empty
      const itinerary = JSON.parse(localStorage.getItem('itinerary')) || [];
      if (itinerary.length === 0) {
        alert('Your itinerary is empty. Please add some places first.');
        return;
      }

      alert('Preparing your itinerary. Please wait...');

      // Hardcoded detailed itinerary for the selected three places
      const hardcodedItinerary = `
Day 1:
Start your day with a hearty breakfast at Cafe Goodluck, one of Pune's oldest and most iconic cafes. Enjoy their famous bun maska and Irani chai while soaking in the nostalgic charm of this historic spot. The quick service and lively atmosphere make it a perfect start to your trip.

Day 2:
Spend your second day exploring the ancient Pataleshwar Cave Temple. This 8th-century rock-cut temple, dedicated to Lord Shiva, is carved out of a single basalt rock. Marvel at the intricate sculptures and experience the serene ambiance of this architectural wonder. It's a peaceful retreat in the heart of the bustling city.

Day 3:
Conclude your trip with a visit to the Shreemant Dagdusheth Halwai Ganpati Temple. This iconic temple, built over 125 years ago, is a symbol of faith and devotion in Pune. Admire the grand Ganesh idol adorned with gold and precious ornaments, and immerse yourself in the spiritual atmosphere. Don't forget to explore the nearby Shaniwar Wada for a glimpse of Pune's rich history.

Nearby Hotels:
1. Hotel Abhijeet New Sweet Home
   Address: 123 MG Road, Pune
   Phone: +91-9876543210

2. Hotel Sahyadri
   Address: 456 FC Road, Pune
   Phone: +91-9123456789

Nearby Taxi Services:
1. Pune City Taxi
   Phone: +91-9988776655

2. Quick Cab Pune
   Phone: +91-8877665544
      `;

      // Notify the user that the itinerary is ready
      alert('Your itinerary is ready! Downloading now...');

      // Create downloadable file
      const blob = new Blob([hardcodedItinerary], { type: 'text/plain' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'itinerary.txt';
      link.click();
    };
  }

  // Add logic for "View 3D Model" button
  function addSketchfabLogic(sketchfabBtn) {
    sketchfabBtn.onclick = function () {
      const sketchfabUrl = 'https://sketchfab.com/models/d0df0ef8afc44c7b98fee535498a175e/embed';
      const modal = document.createElement('div');
      modal.id = 'sketchfab-modal';
      modal.style.position = 'fixed';
      modal.style.top = '0';
      modal.style.left = '0';
      modal.style.width = '100%';
      modal.style.height = '100%';
      modal.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
      modal.style.display = 'flex';
      modal.style.justifyContent = 'center';
      modal.style.alignItems = 'center';
      modal.style.zIndex = '1000';

      const iframe = document.createElement('iframe');
      iframe.src = sketchfabUrl;
      iframe.style.width = '80%';
      iframe.style.height = '80%';
      iframe.style.border = 'none';
      iframe.allow = 'autoplay; fullscreen; xr-spatial-tracking';

      const closeModal = document.createElement('button');
      closeModal.textContent = 'Close';
      closeModal.style.position = 'absolute';
      closeModal.style.top = '10px';
      closeModal.style.right = '10px';
      closeModal.style.padding = '10px';
      closeModal.style.backgroundColor = '#fff';
      closeModal.style.border = 'none';
      closeModal.style.cursor = 'pointer';

      closeModal.onclick = function () {
        document.body.removeChild(modal);
      };

      modal.appendChild(iframe);
      modal.appendChild(closeModal);
      document.body.appendChild(modal);
    };
  }

  // Call the function to add the bottom control buttons
  addBottomControlButtons();
});
