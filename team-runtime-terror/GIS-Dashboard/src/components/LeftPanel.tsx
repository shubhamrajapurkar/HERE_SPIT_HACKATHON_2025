import React from 'react';
import { Eye, EyeOff, Trash2 } from 'lucide-react';
import { FeatureCollection } from 'geojson';

interface Layer {
  id: number;
  name: string;
  type: string;
  visible: boolean;
  color: string;
  data: FeatureCollection;
}

interface LeftPanelProps {
  layers: Layer[];
  selectedLayers: number[];
  onToggleVisibility: (id: number) => void;
  onToggleSelection: (id: number) => void;
  onRemoveLayer: (id: number) => void;
  onLoadSampleData: (type: 'polygon' | 'line' | 'point') => void;
}

const LeftPanel: React.FC<LeftPanelProps> = ({
  layers,
  selectedLayers,
  onToggleVisibility,
  onToggleSelection,
  onRemoveLayer,
  onLoadSampleData
}) => {
  return (
    <div className="space-y-4">
      {/* Sample Data Section */}
      <div className="bg-white rounded-xl shadow-lg p-4 border-l-4" style={{borderColor: '#287094'}}>
        <h3 className="text-lg font-semibold mb-4 text-[#023246]">Sample Data</h3>
        <div className="grid grid-cols-1 gap-2">
          <button
            onClick={() => onLoadSampleData('polygon')}
            className="bg-[#287094] text-white px-4 py-2 rounded-lg hover:bg-[#023246] transition-colors"
          >
            Load Sample Zones
          </button>
          <button
            onClick={() => onLoadSampleData('line')}
            className="bg-[#287094] text-white px-4 py-2 rounded-lg hover:bg-[#023246] transition-colors"
          >
            Load Sample Roads
          </button>
          <button
            onClick={() => onLoadSampleData('point')}
            className="bg-[#287094] text-white px-4 py-2 rounded-lg hover:bg-[#023246] transition-colors"
          >
            Load Sample Buildings
          </button>
        </div>
      </div>

      {/* Layers Section
      <div className="bg-white rounded-xl shadow-lg p-4 border-l-4" style={{borderColor: '#287094'}}>
        <h3 className="text-lg font-semibold mb-4 text-[#023246]">Layers</h3>
        <div className="space-y-2">
          {layers.map(layer => (
            <div
              key={layer.id}
              className="flex items-center justify-between p-2 bg-gray-50 rounded-lg"
            >
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={selectedLayers.includes(layer.id)}
                  onChange={() => onToggleSelection(layer.id)}
                  className="w-4 h-4"
                />
                <span className="text-sm text-gray-800">{layer.name}</span>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => onToggleVisibility(layer.id)}
                  className="p-1 hover:bg-gray-200 rounded text-[#287094]"
                >
                  {layer.visible ? (
                    <Eye className="w-4 h-4" />
                  ) : (
                    <EyeOff className="w-4 h-4" />
                  )}
                </button>
                <button
                  onClick={() => onRemoveLayer(layer.id)}
                  className="p-1 hover:bg-gray-200 rounded text-[#287094]"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div> */}
      {/* Layers Section */}
<div className="bg-white rounded-xl shadow-lg p-4 border-l-4" style={{ borderColor: '#287094' }}>
  <h3 className="text-lg font-semibold mb-4 text-[#023246]">Layers</h3>

  {/* Scrollable list container */}
  <div
    className="space-y-2 overflow-y-auto pr-1"
    style={{
      maxHeight: '88px', // Adjust based on item height (approx. 44px * 5)
      scrollbarWidth: 'thin',
    }}
  >
    <style>
      {`
        .scrollable-layers::-webkit-scrollbar {
          width: 6px;
          background: transparent;
        }
        .scrollable-layers::-webkit-scrollbar-thumb {
          background: #b0b0b0;
          border-radius: 4px;
        }
        .scrollable-layers {
          scrollbar-width: thin;
          scrollbar-color: #b0b0b0 transparent;
        }
      `}
    </style>

    <div className="scrollable-layers">
      {layers.map(layer => (
        <div
          key={layer.id}
          className="flex items-center justify-between p-2 bg-gray-50 rounded-lg"
        >
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={selectedLayers.includes(layer.id)}
              onChange={() => onToggleSelection(layer.id)}
              className="w-4 h-4"
            />
            <span className="text-sm text-gray-800">{layer.name}</span>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => onToggleVisibility(layer.id)}
              className="p-1 hover:bg-gray-200 rounded text-[#287094]"
            >
              {layer.visible ? (
                <Eye className="w-4 h-4" />
              ) : (
                <EyeOff className="w-4 h-4" />
              )}
            </button>
            <button
              onClick={() => onRemoveLayer(layer.id)}
              className="p-1 hover:bg-gray-200 rounded text-[#287094]"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      ))}
    </div>
  </div>
</div>

    </div>
  );
};

export default LeftPanel;