import React, { useEffect, useRef, useState } from 'react';
import H from '@here/maps-api-for-javascript';

const refData = {
  "shettys corner": {
    lat: 19.124838,
    lng: 72.837949,
  },
  "radha krishna": {
    lat: 19.1152154,
    lng: 72.8424089,
  }
};

const RoutePlanner = () => {
  const mapRef = useRef(null);
  const map = useRef(null);
  const platform = useRef(null);
  const [source, setSource] = useState('');
  const [destination, setDestination] = useState('');
  const [numStops, setNumStops] = useState(1);
  const [routeInfo, setRouteInfo] = useState(null);

  useEffect(() => {
    // Initialize the map
    if (!map.current) {
      platform.current = new H.service.Platform({
        apikey: import.meta.env.VITE_HERE_API_KEY
      });

      const defaultLayers = platform.current.createDefaultLayers();

      map.current = new H.Map(
        mapRef.current,
        defaultLayers.vector.normal.map,
        {
          center: { lat: 19.1152154, lng: 72.8424089 }, // Center on Mumbai area
          zoom: 17,
          pixelRatio: window.devicePixelRatio || 1
        }
      );

      // Enable map interaction (pan, zoom, etc.)
      new H.mapevents.Behavior(new H.mapevents.MapEvents(map.current));

      // Add UI controls
      H.ui.UI.createDefault(map.current, defaultLayers);

      // Make the map responsive
      window.addEventListener('resize', () => map.current.getViewPort().resize());
    }
  }, []);

  const formatDuration = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
  };

  const formatDistance = (meters) => {
    const km = meters / 1000;
    return `${km.toFixed(1)} km`;
  };

  const calculateRoute = async () => {
    if (!source || !destination) {
      alert('Please enter both source and destination');
      return;
    }

    if (!platform.current || !map.current) {
      alert('Map is not initialized yet. Please try again in a moment.');
      return;
    }

    try {
      // Clear existing route if any
      map.current.removeObjects(map.current.getObjects());

      // Get coordinates for source and destination
      const geocoder = platform.current.getSearchService();

      const [sourceResult, destResult] = await Promise.all([
        new Promise((resolve, reject) => {
          geocoder.geocode({ q: source, limit: 1 }, (result) => {
            if (result.items && result.items.length > 0) {
              resolve(result.items[0].position);
            } else {
              reject(new Error('Source location not found'));
            }
          }, reject);
        }),
        new Promise((resolve, reject) => {
          geocoder.geocode({ q: destination, limit: 1 }, (result) => {
            if (result.items && result.items.length > 0) {
              resolve(result.items[0].position);
            } else {
              reject(new Error('Destination location not found'));
            }
          }, reject);
        })
      ]);

      // Get restaurant coordinates - randomly select based on number of stops
      const restaurants = Object.values(refData);
      const selectedRestaurants = restaurants.slice(0, Math.min(2, numStops));

      // Calculate routes through all selected restaurants
      const router = platform.current.getRoutingService(null, 8);

      let allPoints = [sourceResult, ...selectedRestaurants.map(r => ({ lat: r.lat, lng: r.lng })), destResult];
      let routeSegments = [];
      let totalDistance = 0;
      let totalDuration = 0;
      let group = new H.map.Group();

      // Calculate routes between consecutive points
      for (let i = 0; i < allPoints.length - 1; i++) {
        const start = allPoints[i];
        const end = allPoints[i + 1];

        const route = await new Promise((resolve, reject) => {
          router.calculateRoute({
            routingMode: 'fast',
            transportMode: 'car',
            origin: `${start.lat},${start.lng}`,
            destination: `${end.lat},${end.lng}`,
            return: 'polyline,summary'
          }, resolve, reject);
        });

        const routeShape = route.routes[0].sections[0].polyline;
        const lineString = H.geo.LineString.fromFlexiblePolyline(routeShape);
        const routeLine = new H.map.Polyline(lineString, {
          style: {
            strokeColor: '#0066FF',
            lineWidth: 6,
            lineCap: 'round'
          }
        });
        group.addObject(routeLine);

        routeSegments.push({
          distance: route.routes[0].sections[0].summary.length,
          duration: route.routes[0].sections[0].summary.duration
        });

        totalDistance += route.routes[0].sections[0].summary.length;
        totalDuration += route.routes[0].sections[0].summary.duration;
      }

      // Add markers
      allPoints.forEach(point => {
        group.addObject(new H.map.Marker(point));
      });

      // Add all objects to the map
      map.current.addObject(group);

      // Set route info
      setRouteInfo({
        totalDistance,
        totalDuration,
        segments: routeSegments
      });

      // Zoom to show all objects
      map.current.getViewModel().setLookAtData({
        bounds: group.getBoundingBox()
      });

    } catch (error) {
      console.error('Error:', error);
      alert(error.message || 'Error calculating route. Please check your inputs and try again.');
    }
  };

  return (
    <div className="h-screen flex flex-col">
      <div className="p-4 bg-white shadow-md">
        <div className="flex flex-col gap-4 max-w-4xl mx-auto">
          <div className="flex gap-4">
            <input
              type="text"
              placeholder="Enter source location"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              className="flex-1 p-2 border rounded"
            />
            <select
              value={numStops}
              onChange={(e) => setNumStops(parseInt(e.target.value))}
              className="p-2 border rounded bg-white"
            >
              {[...Array(10)].map((_, i) => (
                <option key={i + 1} value={i + 1}>
                  {i + 1} Stop{i !== 0 ? 's' : ''}
                </option>
              ))}
            </select>
            <input
              type="text"
              placeholder="Enter destination"
              value={destination}
              onChange={(e) => setDestination(e.target.value)}
              className="flex-1 p-2 border rounded"
            />
            <button
              onClick={calculateRoute}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Calculate Route
            </button>
          </div>

          {/* Route information display */}
          {routeInfo && (
            <div className="mt-4 grid grid-cols-2 gap-4">
              <div className="p-3 rounded border border-blue-500 bg-blue-50">
                <div className="font-semibold">Total Distance</div>
                <div className="text-sm">
                  {formatDistance(routeInfo.totalDistance)}
                </div>
              </div>
              <div className="p-3 rounded border border-blue-500 bg-blue-50">
                <div className="font-semibold">Total Duration</div>
                <div className="text-sm">
                  {formatDuration(routeInfo.totalDuration)}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
      <div ref={mapRef} className="flex-1" />
    </div>
  );
};

export default RoutePlanner;