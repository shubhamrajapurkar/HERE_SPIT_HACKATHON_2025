const API_URL = 'http://localhost:5000/api/map-data';

export const getMapData = async () => {
  const response = await fetch(API_URL);
  if (!response.ok) {
    throw new Error('Failed to fetch map data');
  }
  return response.json();
};

export const addLocation = async (location) => {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(location),
  });
  if (!response.ok) {
    throw new Error('Failed to add location');
  }
  return response.json();
};