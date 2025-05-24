import React from 'react';
import { Settings, Circle, Plus, Triangle, Minus, Building } from 'lucide-react';

type OperationType = 'buffer' | 'union' | 'intersection' | 'difference' | 'spatialJoin';

interface OperationsSectionProps {
  performOperation: (operation: OperationType) => void;
  selectedLayers: any[];
}

const OperationsSection: React.FC<OperationsSectionProps> = ({ 
  performOperation,
  selectedLayers
}) => {
  const operations: { key: OperationType; label: string; icon: any; color: string }[] = [
    { key: 'buffer', label: 'Buffer', icon: Circle, color: '#287094' },
    { key: 'union', label: 'Union', icon: Plus, color: '#287094' },
    { key: 'intersection', label: 'Intersection', icon: Triangle, color: '#023246' },
    { key: 'difference', label: 'Difference', icon: Minus, color: '#023246' },
    { key: 'spatialJoin', label: 'Spatial Join', icon: Building, color: '#287094' }
  ];

  return (
    <div className="bg-white rounded-xl shadow-lg p-4 border-l-4 mt-4" style={{borderColor: '#287094'}}>
      <h3 className="text-lg font-semibold mb-4 text-[#023246] flex items-center">
        <Settings className="w-5 h-5 mr-2 text-[#287094]" />
        GIS Operations
      </h3>
      <div className="grid grid-cols-1 gap-2">
        {operations.map((op) => {
          const IconComponent = op.icon;
          // Enable buffer if at least 1 layer is selected, others need 2
          const isDisabled =
            op.key === 'buffer'
              ? selectedLayers.length < 1
              : selectedLayers.length < 2;
          return (
            <button
              key={op.key}
              onClick={() => performOperation(op.key)}
              disabled={isDisabled}
              className="text-white py-2 px-4 rounded-lg flex items-center space-x-2 transition-all duration-200 font-bold hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
              style={{backgroundColor: op.color}}
            >
              <IconComponent className="w-4 h-4" />
              <span>{op.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default OperationsSection;