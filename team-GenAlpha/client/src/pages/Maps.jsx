import React, { useEffect, useRef, useState } from 'react';
import { 
  Download, 
  Settings,
  Layers,
  MapPin,
  Route,
  TrendingUp,
  Map
} from 'lucide-react';

const Maps = () => {
  const mapRef = useRef(null);
  const displayRef = useRef(null);
  const [mapInitialized, setMapInitialized] = useState(false);

  // Sample GeoJSON data
  const geoJsonData = {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        properties: {},
        geometry: {
          type: "Point",
          coordinates: [8.53422, 50.16212]
        }
      },
      {
        type: "Feature",
        properties: {},
        geometry: {
          type: "LineString",
          coordinates: [
            [8.53422, 50.16212],
            [8.53450, 50.16230],
            [8.53500, 50.16210]
          ]
        }
      }
    ]
  };

  useEffect(() => {
    // Check if XYZMaps is loaded
    const checkXYZMaps = () => {
      if (window.XYZMaps && mapRef.current && !displayRef.current) {
        initializeMap();
      } else {
        setTimeout(checkXYZMaps, 100);
      }
    };

    const initializeMap = () => {
      try {
        console.log('Initializing XYZ Map...');
        
        // Base map layer configuration
        const baseLayer = {
          name: 'base-layer',
          style: {
            styleGroups: {
              tile: [{
                zIndex: 0,
                type: 'Rect',
                fill: '#F5F5F5'
              }]
            },
            assign: () => 'tile'
          },
          url: 'https://xyz.api.here.com/tiles/osmbase/512/all/{z}/{x}/{y}.mvt?access_token=6JDMdJ79eaBnqgPSNPj7V-Ut6nz1XSyKfjQSjF4SgnM'
        };

        // GeoJSON layer configuration
        const geoJsonLayer = {
          name: 'geojson-layer',
          style: {
            styleGroups: {
              Point: [{
                zIndex: 1,
                type: 'Circle',
                radius: 8,
                fill: '#FF0000',
                stroke: '#FFFFFF',
                strokeWidth: 2
              }],
              LineString: [{
                zIndex: 1,
                type: 'Line',
                stroke: '#3366FF',
                strokeWidth: 3
              }]
            },
            assign: (feature) => feature.geometry.type
          },
          data: geoJsonData
        };

        // Initialize the map
        displayRef.current = new window.XYZMaps.MapDisplay(mapRef.current, {
          zoomLevel: 18,
          center: {
            longitude: 8.53422,
            latitude: 50.16212
          },
          layers: [baseLayer, geoJsonLayer],
          behaviors: {
            zoom: true,
            drag: true
          }
        });

        console.log('Map initialized successfully');
        setMapInitialized(true);

      } catch (error) {
        console.error('Error initializing XYZ Map:', error);
      }
    };

    checkXYZMaps();

    return () => {
      if (displayRef.current) {
        console.log('Cleaning up map');
        displayRef.current.destroy();
        displayRef.current = null;
      }
    };
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">Generated Maps</h2>
        <div className="flex space-x-3">
          <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            <Download className="h-4 w-4" />
            <span>Export</span>
          </button>
          <button className="flex items-center space-x-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700">
            <Settings className="h-4 w-4" />
            <span>Settings</span>
          </button>
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900">Interactive Map View</h3>
            <div className="flex items-center space-x-2">
              <select className="px-3 py-2 border border-gray-300 rounded-lg text-sm">
                <option>Default View</option>
                <option>Satellite View</option>
                <option>Street View</option>
              </select>
              <button className="p-2 border border-gray-300 rounded-lg hover:bg-gray-100">
                <Layers className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
        
        <div 
          ref={mapRef} 
          className="h-96 w-full"
          style={{ 
            minHeight: '384px',
            position: 'relative',
            overflow: 'hidden',
            backgroundColor: mapInitialized ? 'transparent' : '#F5F5F5'
          }}
        >
          {!mapInitialized && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <Map className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">Loading map...</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <MapPin className="h-6 w-6 text-blue-600" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">Points of Interest</h3>
              <p className="text-2xl font-bold text-blue-600">2,847</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-green-100 rounded-lg">
              <Route className="h-6 w-6 text-green-600" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">Road Segments</h3>
              <p className="text-2xl font-bold text-green-600">12,456</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-purple-100 rounded-lg">
              <TrendingUp className="h-6 w-6 text-purple-600" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">Accuracy Score</h3>
              <p className="text-2xl font-bold text-purple-600">96.8%</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Maps;