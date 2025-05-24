import axios from 'axios';

// Replace with your actual API base URL
const API_BASE_URL = ' https://0016-103-104-226-58.ngrok-free.app'; // Update this with your actual backend URL

class HereApiService {
  // Calculate route between two points
  static async calculateRoute(source, destination, transportMode = 'car') {
    try {
      console.log(`Calculating route from ${source} to ${destination} with transport mode ${transportMode}`);
      
      const response = await axios.post(`${API_BASE_URL}/routes/calulate`, {
        source,
        destination,
        transportMode
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to calculate route');
    }
  }

  // Get alternative routes
  static async getAlternativeRoutes(source, destination, transportMode = 'car', alternatives = 3) {
    try {
      // console.log(`Fetching alternative routes from ${source} to ${destination} with transport mode ${transportMode} and ${alternatives} alternatives`);
      console.log(source);
      console.log(destination);
      console.log(`${API_BASE_URL}/routes/alternate`)
      const response = await axios.post(`${API_BASE_URL}/routes/alternate`, {
        source,
        destination,
        transportMode,
        alternatives
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to get alternative routes');
    }
  }

  // Get traffic incidents
  static async getTrafficIncidents(latitude, longitude, radius = 5000) {
    try {
      const response = await axios.get(`${API_BASE_URL}/routes/traffic`, {
        params: {
          latitude,
          longitude,
          radius
        }
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to get traffic incidents');
    }
  }

  // Get emergency route
  static async getEmergencyRoute(userLocation, type = 'hospital') {
    try {
      const response = await axios.post(`${API_BASE_URL}/routes/emergency`, {
        userLocation,
        type
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to get emergency route');
    }
  }
}

export default HereApiService;