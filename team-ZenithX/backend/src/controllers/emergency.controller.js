import axios from 'axios';
const HERE_API_KEY = "9Dpxsv-LXOYn_WzclZVeaRi8-R2WOJkG75xWaAW01pk";

// Emergency route controller
async function emergencyRoute(req, res) {
  try {
    const { userLocation, type = 'hospital' } = req.body;
    if (!userLocation || !userLocation.lat || !userLocation.lng) {
      return res.status(400).json({
        success: false,
        message: 'User location with lat and lng is required'
      });
    }

    // Use HERE Geocoding & Search API to find the nearest emergency facility
    const categoryMap = {
      hospital: 'hospital-health-care-facility',
      police: 'police-station',
      fire: 'fire-station'
    };
    const category = categoryMap[type] || categoryMap.hospital;

    // Find the nearest place
     const placesResp = await axios.get('https://discover.search.hereapi.com/v1/discover', {
      params: {
        apiKey: HERE_API_KEY,
        at: `${userLocation.lat},${userLocation.lng}`,
        q: type, // e.g., 'hospital', 'police', 'fire'
        limit: 1
      }
    });

    if (!placesResp.data.items || placesResp.data.items.length === 0) {
      return res.status(404).json({
        success: false,
        message: `No nearby ${type} found`
      });
    }

    const destination = placesResp.data.items[0].position;

    // Calculate route from user to the nearest facility
    const routeResp = await axios.get('https://router.hereapi.com/v8/routes', {
      params: {
        apiKey: HERE_API_KEY,
        transportMode: 'car',
        origin: `${userLocation.lat},${userLocation.lng}`,
        destination: `${destination.lat},${destination.lng}`,
        return: 'polyline,summary'
      }
    });

    return res.status(200).json({
      success: true,
      facility: placesResp.data.items[0],
      route: routeResp.data
    });
  } catch (error) {
    console.error('Emergency route error:', error.message);
    return res.status(500).json({
      success: false,
      message: 'Failed to calculate emergency route',
      error: error.message
    });
  }
}

export {
  emergencyRoute,
}