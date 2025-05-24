import React from 'react';
import { Layers, Eye, EyeOff, Trash2 } from 'lucide-react';
import { FeatureCollection } from 'geojson';

interface Layer {
  id: number;
  name: string;
  type: string;
  visible: boolean;
  color: string;
  data: FeatureCollection;
}

interface LayersSectionProps {
  layers: Layer[];
  selectedLayers: number[];
  toggleLayerVisibility: (id: number) => void;
  toggleLayerSelection: (id: number) => void;
  removeLayer: (id: number) => void;
}

const LayersSection: React.FC<LayersSectionProps> = ({
  layers,
  selectedLayers,
  toggleLayerVisibility,
  toggleLayerSelection,
  removeLayer
}) => {
  return (
    <div
      className="bg-white rounded-xl shadow-lg p-6 border-l-4"
      style={{
        borderColor: '#287094',
        maxHeight: '80vh',        // Limit the section height to 80% of viewport
        overflowY: 'auto',        // Make the entire section scrollable if needed
        minHeight: '200px',       // Optional: minimum height for aesthetics
        width: '100%',            // Ensure it doesn't overflow horizontally
        boxSizing: 'border-box'
      }}
    >
      <h3 className="text-xl font-bold mb-4 flex items-center" style={{color: '#023246'}}>
        <Layers className="w-6 h-6 mr-2" style={{color: '#287094'}} />
        Layers ({layers.length})
      </h3>
      <div className="space-y-2">
        {layers.length === 0 ? (
          <p className="text-center py-4" style={{color: '#D4D4CE'}}>No layers loaded</p>
        ) : (
          layers.map((layer) => (
            <div
              key={layer.id}
              className={`p-3 rounded-lg border-2 transition-all duration-200 ${
                selectedLayers.includes(layer.id)
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200'
              }`}
              style={{
                backgroundColor: selectedLayers.includes(layer.id) ? 'rgba(40, 112, 148, 0.1)' : '#F6F6F6',
                borderColor: selectedLayers.includes(layer.id) ? '#287094' : '#D4D4CE'
              }}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={selectedLayers.includes(layer.id)}
                    onChange={() => toggleLayerSelection(layer.id)}
                    className="w-4 h-4"
                    style={{accentColor: '#287094'}}
                  />
                  <div
                    className="w-4 h-4 rounded-full border-2"
                    style={{ backgroundColor: layer.color, borderColor: '#D4D4CE' }}
                  ></div>
                  <span className="font-medium text-sm truncate" style={{color: '#023246'}}>{layer.name}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <button
                    onClick={() => toggleLayerVisibility(layer.id)}
                    className="p-1 rounded transition-colors"
                    style={{
                      color: layer.visible ? '#287094' : '#D4D4CE'
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'rgba(212, 212, 206, 0.3)')}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                  >
                    {layer.visible ? (
                      <Eye className="w-4 h-4" />
                    ) : (
                      <EyeOff className="w-4 h-4" />
                    )}
                  </button>
                  <button
                    onClick={() => removeLayer(layer.id)}
                    className="p-1 rounded transition-colors"
                    style={{color: '#023246'}}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'rgba(2, 50, 70, 0.1)')}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default LayersSection;