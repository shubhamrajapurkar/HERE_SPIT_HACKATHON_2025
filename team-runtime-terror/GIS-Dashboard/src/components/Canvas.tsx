import React, { useEffect, useRef } from 'react';
import { ZoomIn, ZoomOut, Map, RotateCcw } from 'lucide-react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet default marker icons
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface CanvasProps {
  canvasRef: React.RefObject<HTMLDivElement>;
  transform: { x: number; y: number; scale: number };
  coordinates: { x: number; y: number };
  setCoordinates: React.Dispatch<React.SetStateAction<{ x: number; y: number }>>;
  zoomIn: () => void;
  zoomOut: () => void;
  resetView: () => void;
  layers: any[];
}

const Canvas: React.FC<CanvasProps> = ({
  canvasRef,
  transform,
  coordinates,
  setCoordinates,
  zoomIn,
  zoomOut,
  resetView,
  layers
}) => {
  const mapRef = useRef<L.Map | null>(null);
  const geoJsonLayersRef = useRef<L.GeoJSON[]>([]);

  useEffect(() => {
    if (!mapRef.current && canvasRef.current) {
      // Initialize map
      mapRef.current = L.map(canvasRef.current).setView([0, 0], 2);
      
      // Add tile layer
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      }).addTo(mapRef.current);

      // Add click handler
      mapRef.current.on('click', (e: L.LeafletMouseEvent) => {
        setCoordinates({ x: e.latlng.lng, y: e.latlng.lat });
      });
    }

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [canvasRef, setCoordinates]);

  // Update map view when transform changes
  useEffect(() => {
    if (mapRef.current) {
      mapRef.current.setView([transform.y, transform.x], transform.scale * 10);
    }
  }, [transform]);

  // Update GeoJSON layers when layers prop changes
  useEffect(() => {
    if (!mapRef.current) return;

    // Remove old layers
    geoJsonLayersRef.current.forEach(layer => layer.remove());
    geoJsonLayersRef.current = [];

    // Add new layers
    layers.forEach(layer => {
      if (layer.visible && layer.data) {
        const geoJsonLayer = L.geoJSON(layer.data, {
          style: {
            color: layer.color,
            weight: 2,
            fillOpacity: 0.4
          }
        }).addTo(mapRef.current!);
        geoJsonLayersRef.current.push(geoJsonLayer);
      }
    });
  }, [layers]);

  return (
    <div className="col-span-6 bg-white rounded-xl shadow-lg overflow-hidden border-2" style={{borderColor: '#D4D4CE'}}>
      <div className="relative h-full">
        <div
          ref={canvasRef}
          className="w-full h-full"
          style={{ minHeight: '675px',minWidth : '650px' ,backgroundColor: '#F6F6F6' }}
        />
        
        {/* Coordinate Display */}
        <div className="absolute top-4 left-4 px-3 py-2 rounded-lg font-mono text-sm text-white" style={{backgroundColor: 'rgba(2, 50, 70, 0.8)'}}>
          X: {coordinates.x.toFixed(2)}, Y: {coordinates.y.toFixed(2)}
        </div>
        
        {/* Zoom Controls */}
        <div className="absolute top-4 right-4 flex flex-col space-y-2">
          <button
            onClick={zoomIn}
            className="bg-white p-3 rounded-lg shadow-lg border transition-colors hover:opacity-90"
            style={{borderColor: '#D4D4CE'}}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#F6F6F6'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'white'}
          >
            <ZoomIn className="w-5 h-5" style={{color: '#023246'}} />
          </button>
          <button
            onClick={zoomOut}
            className="bg-white p-3 rounded-lg shadow-lg border transition-colors hover:opacity-90"
            style={{borderColor: '#D4D4CE'}}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#F6F6F6'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'white'}
          >
            <ZoomOut className="w-5 h-5" style={{color: '#023246'}} />
          </button>
          <button
            onClick={resetView}
            className="bg-white p-3 rounded-lg shadow-lg border transition-colors hover:opacity-90"
            style={{borderColor: '#D4D4CE'}}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#F6F6F6'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'white'}
          >
            <RotateCcw className="w-5 h-5" style={{color: '#023246'}} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Canvas;