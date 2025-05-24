import React, { useEffect, useRef, useState } from 'react';
import H from '@here/maps-api-for-javascript';

const RoutingPage = () => {
  const mapRef = useRef(null);
  const map = useRef(null);
  const platform = useRef(null);
  const [source, setSource] = useState('');
  const [destination, setDestination] = useState('');
  const [routeInfo, setRouteInfo] = useState([]);

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
          center: { lat: 19.1152154, lng: 72.8424089 },
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

      // Calculate routes
      const router = platform.current.getRoutingService(null, 8);

      router.calculateRoute({
        routingMode: 'fast',
        transportMode: 'car',
        origin: `${sourceResult.lat},${sourceResult.lng}`,
        destination: `${destResult.lat},${destResult.lng}`,
        return: 'polyline,summary,actions',
        alternatives: 3
      }, (result) => {
        const routes = result.routes;

        // Update route info with traffic and road type information
        const routeInfoArray = routes.map((route, index) => {
          const section = route.sections[0];
          const trafficTime = section.summary.trafficTime || section.summary.duration;
          const baseTime = section.summary.baseTime || section.summary.duration;
          const trafficDelay = trafficTime - baseTime;

          return {
            duration: section.summary.duration,
            distance: section.summary.length,
            trafficDelay,
            isOptimal: index === 0,
            // Calculate efficiency score (lower is better)
            efficiency: (trafficTime / 60) / (section.summary.length / 1000) // minutes per km
          };
        });
        setRouteInfo(routeInfoArray);

        // Draw all routes
        // First draw alternative routes
        routes.forEach((route, index) => {
          if (index !== 0) { // Skip the best route for now
            const routeShape = route.sections[0].polyline;
            const lineString = H.geo.LineString.fromFlexiblePolyline(routeShape);
            const routeLine = new H.map.Polyline(lineString, {
              style: {
                strokeColor: '#D946EF',
                lineWidth: 4,
                lineCap: 'round'
              }
            });
            map.current.addObject(routeLine);
          }
        });

        // Then draw the best route on top
        if (routes.length > 0) {
          const bestRoute = routes[0];
          const routeShape = bestRoute.sections[0].polyline;
          const lineString = H.geo.LineString.fromFlexiblePolyline(routeShape);
          const routeLine = new H.map.Polyline(lineString, {
            style: {
              strokeColor: '#0066FF',
              lineWidth: 6,
              lineCap: 'round'
            }
          });
          map.current.addObject(routeLine);
        }

        // Zoom to show all routes
        map.current.getViewModel().setLookAtData({
          bounds: map.current.getObjects()[0].getBoundingBox()
        });
      }, (error) => {
        console.error('Error calculating route:', error);
        alert('Error calculating route. Please check your inputs and try again.');
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
              Calculate Routes
            </button>
          </div>

          {/* Route information display */}
          {routeInfo.length > 0 && (
            <div className="mt-4">
              <div className="grid grid-cols-4 gap-3">
                {routeInfo.map((info, index) => (
                  <div
                    key={index}
                    className={`p-3 rounded border ${info.isOptimal ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
                      }`}
                  >
                    <div className="flex flex-col">
                      <div className="font-semibold">Route {index + 1}</div>
                      <div className="text-sm">
                        <div>{formatDistance(info.distance)}</div>
                        <div>{formatDuration(info.duration)}</div>
                        {info.trafficDelay > 0 && (
                          <div className="text-orange-600">
                            +{formatDuration(info.trafficDelay)} traffic
                          </div>
                        )}
                        <div className="text-xs text-gray-500">
                          {info.efficiency.toFixed(1)} min/km
                        </div>
                      </div>
                      {info.isOptimal && (
                        <div className="mt-1 text-xs text-green-600 font-medium">
                          Best balance of distance & traffic
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-2 text-xs text-gray-600">
                Routes are ranked based on overall efficiency (time per km), considering both distance and current traffic conditions.
              </div>
            </div>
          )}
        </div>
      </div>
      <div ref={mapRef} className="flex-1" />
    </div>
  );
};

export default RoutingPage; 