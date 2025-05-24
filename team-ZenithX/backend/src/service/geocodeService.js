import  axios from 'axios';
// import config from ('../config';

export const HERE_API_KEY = "9Dpxsv-LXOYn_WzclZVeaRi8-R2WOJkG75xWaAW01pk"

async function  geocode(query) {
    try {
      if (!query) {
        throw new Error('Query parameter is required');
      }

      const response = await axios.get('https://geocode.search.hereapi.com/v1/geocode', {
        params: {
          apiKey: HERE_API_KEY,
          q: query,
          limit: 5
        }
      });

      if (!response.data.items || response.data.items.length === 0) {
        return {
          success: false,
          message: 'No location found for the given query'
        };
      }

      const results = response.data.items.map(item => ({
        title: item.title,
        address: item.address,
        position: item.position, // Contains lat and lng
        mapView: item.mapView // Bounding box of the location
      }));
      console.log('Geocoding results:', results);
      
      return {
        success: true,
        data: results
      };
    } catch (error) {
      console.error('Geocoding error:', error.message);
      throw new Error(`Failed to geocode location: ${error.message}`);
    }
  }

  async  function  reverseGeocode(lat, lng) {
    try {
      if (!lat || !lng) {
        throw new Error('Latitude and longitude are required');
      }

      const response = await axios.get('https://revgeocode.search.hereapi.com/v1/revgeocode', {
        params: {
          apiKey: HERE_API_KEY,
          at: `${lat},${lng}`,
          lang: 'en-US'
        }
      });

      // If no results found
      if (!response.data.items || response.data.items.length === 0) {
        return {
          success: false,
          message: 'No address found for the given coordinates'
        };
      }

      const results = response.data.items.map(item => ({
        title: item.title,
        address: item.address,
        position: item.position,
        distance: item.distance // Distance in meters from the query point
      }));

      return {
        success: true,
        data: results
      };
    } catch (error) {
      console.error('Reverse geocoding error:', error.message);
      throw new Error(`Failed to reverse geocode: ${error.message}`);
    }
  }

  async  function autocomplete(query, options = {}) {
    try {
      if (!query) {
        throw new Error('Query parameter is required');
      }

      const params = {
        apiKey: HERE_API_KEY,
        q: query,
        limit: options.limit || 5,
        lang: options.lang || 'en-US'
      };

      // Add country restriction if provided
      if (options.countries) {
        params.in = `countryCode:${options.countries}`;
      }

      const response = await axios.get('https://autocomplete.search.hereapi.com/v1/autocomplete', {
        params
      });

      // Format the response
      const suggestions = response.data.items.map(item => ({
        title: item.title,
        id: item.id,
        resultType: item.resultType,
        address: item.address,
        highlights: item.highlights
      }));

      return {
        success: true,
        data: suggestions
      };
    } catch (error) {
      console.error('Autocomplete error:', error.message);
      throw new Error(`Failed to get autocomplete suggestions: ${error.message}`);
    }
  }


  geocode('Pune');
    

export {
    geocode,
    reverseGeocode,
    autocomplete,
}