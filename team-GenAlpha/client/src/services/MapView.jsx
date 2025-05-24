import React, { useEffect, useRef } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

mapboxgl.accessToken = 'YOUR_MAPBOX_ACCESS_TOKEN';

const MapView = ({ mapData, onSelectLocation }) => {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const markers = useRef([]);

  useEffect(() => {
    if (map.current) return; // initialize map only once
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/streets-v11',
      center: [77.2090, 28.6139], // Default to Delhi
      zoom: 12
    });
  }, []);

  useEffect(() => {
    if (!map.current || !mapData.length) return;

    // Clear existing markers
    markers.current.forEach(marker => marker.remove());
    markers.current = [];

    // Add new markers
    mapData.forEach(location => {
      const el = document.createElement('div');
      el.className = 'marker';
      el.style.backgroundImage = `url(${getMarkerIcon(location.type)})`;
      el.style.width = '30px';
      el.style.height = '30px';
      el.style.backgroundSize = 'cover';
      el.style.cursor = 'pointer';

      const marker = new mapboxgl.Marker(el)
        .setLngLat([location.lng, location.lat])
        .addTo(map.current);

      marker.getElement().addEventListener('click', () => {
        onSelectLocation(location);
        map.current.flyTo({
          center: [location.lng, location.lat],
          zoom: 15
        });
      });

      markers.current.push(marker);
    });
  }, [mapData, onSelectLocation]);

  const getMarkerIcon = (type) => {
    switch(type) {
      case 'shop': return '/shop-icon.png';
      case 'restaurant': return '/restaurant-icon.png';
      default: return '/default-icon.png';
    }
  };

  return <div ref={mapContainer} className="map-container" />;
};

export default MapView;