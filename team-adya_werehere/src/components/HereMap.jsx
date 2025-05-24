import { useEffect, useRef, useState } from 'react';
import { useHere } from '../context/HereContext';
import restaurantData from '../../data/restaurant_review.json';
import Popup from './Popup';

const HereMap = ({ zoom = 13, position = { lat: 19.1174, lng: 72.8484 } }) => {
  const mapRef = useRef(null);
  const map = useRef(null);
  const { platform } = useHere();
  const [markers, setMarkers] = useState([]);
  const [popupOpen, setPopupOpen] = useState(false);
  const [selectedRestaurant, setSelectedRestaurant] = useState(null);
  const [popupPosition, setPopupPosition] = useState({ x: 0, y: 0 });

  const getDietaryColor = (restaurant) => {
    const { dietary_options } = restaurant;
    
    if (dietary_options?.jain) return '#0000FF'; // Blue for Jain
    if (dietary_options?.vegan) return '#90EE90'; // Light green for Vegan
    if (dietary_options?.vegetarian) return '#008000'; // Green for Vegetarian
    return '#FF0000'; // Red for Non-vegetarian
  };

  const createMarkerIcon = (restaurant) => {
    const color = getDietaryColor(restaurant);
    const rating = restaurant.rating || 0;
    const size = 32;

    const svgIcon = `
      <svg width="${size}" height="${size}" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur in="SourceAlpha" stdDeviation="2"/>
            <feOffset dx="0" dy="2"/>
            <feComponentTransfer>
              <feFuncA type="linear" slope="0.4"/>
            </feComponentTransfer>
            <feMerge>
              <feMergeNode/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        <circle cx="16" cy="16" r="14" fill="${color}" filter="url(#shadow)" stroke="white" stroke-width="2"/>
        <text x="16" y="20" text-anchor="middle" fill="white" font-size="10" font-weight="bold">${rating.toFixed(1)}</text>
      </svg>
    `;

    return new H.map.Icon(
      `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgIcon)}`,
      { size: { w: size, h: size }, anchor: { x: size/2, y: size/2 } }
    );
  };

  const addRestaurantMarkers = () => {
    if (!map.current) return;

    // Clear existing markers
    markers.forEach(marker => map.current.removeObject(marker));

    const newMarkers = restaurantData
      .filter(r => r.coords && r.coords.lat !== 0 && r.coords.lng !== 0)
      .map((restaurant) => {
        const marker = new H.map.Marker(
          { lat: restaurant.coords.lat, lng: restaurant.coords.lng },
          { icon: createMarkerIcon(restaurant) }
        );

        // Add click event
        marker.addEventListener('tap', (evt) => {
          const screenPoint = map.current.geoToScreen(evt.target.getGeometry());
          const mapRect = mapRef.current.getBoundingClientRect();

          setSelectedRestaurant(restaurant);
          setPopupOpen(true);
          setPopupPosition({
            x: screenPoint.x - mapRect.left,
            y: screenPoint.y - mapRect.top
          });
        });

        return marker;
      });

    newMarkers.forEach(marker => map.current.addObject(marker));
    setMarkers(newMarkers);
  };

  // Initialize map
  useEffect(() => {
    if (!mapRef.current || !platform || map.current) return;

    const defaultLayers = platform.createDefaultLayers();
    
    const newMap = new H.Map(
      mapRef.current,
      defaultLayers.vector.normal.map,
      {
        zoom,
        center: position,
        pixelRatio: window.devicePixelRatio || 1
      }
    );

    new H.mapevents.Behavior(new H.mapevents.MapEvents(newMap));
    const ui = H.ui.UI.createDefault(newMap, defaultLayers);
    
    map.current = newMap;
    addRestaurantMarkers();
  }, [platform]);

  // Update markers when map is moved
  useEffect(() => {
    if (map.current && position) {
      map.current.setCenter(position);
    }
  }, [position]);

  return (
    <div style={{ position: 'relative' }}>
      <div
        ref={mapRef}
        style={{
          width: "100%",
          height: "calc(100vh - 64px)",
          borderRadius: "12px",
          overflow: "hidden",
          boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
          marginTop: "16px"
        }}
      />
      {popupOpen && selectedRestaurant && (
        <Popup
          isOpen={popupOpen}
          onClose={() => setPopupOpen(false)}
          anchorPosition={popupPosition}
          sideContent={
            <>
              <h4>Details</h4>
              <p><b>Type:</b> {
                selectedRestaurant.dietary_options?.jain ? 'Jain' :
                selectedRestaurant.dietary_options?.vegan ? 'Vegan' :
                selectedRestaurant.dietary_options?.vegetarian ? 'Vegetarian' :
                'Non-vegetarian'
              }</p>
              <p><b>Rating:</b> {selectedRestaurant.rating}/5</p>
              <p><b>Quality:</b> {selectedRestaurant.quality}/10</p>
              <p><b>Hygiene:</b> {selectedRestaurant.hygiene}</p>
              {selectedRestaurant.cuisines?.length > 0 && (
                <p><b>Cuisines:</b> {selectedRestaurant.cuisines.join(', ')}</p>
              )}
            </>
          }
        >
          <>
            <h2>{selectedRestaurant.name}</h2>
            <p>{selectedRestaurant.location}</p>
          </>
        </Popup>
      )}
    </div>
  );
};

export default HereMap;