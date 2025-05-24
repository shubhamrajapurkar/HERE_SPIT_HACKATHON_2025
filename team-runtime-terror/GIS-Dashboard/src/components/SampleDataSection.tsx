import React from 'react';
import { Database } from 'lucide-react';

interface SampleDataSectionProps {
  loadSampleData: (type: 'polygons' | 'lines' | 'buildings') => void;
}

const SampleDataSection: React.FC<SampleDataSectionProps> = ({ loadSampleData }) => {
  const sampleDataTypes = [
    { type: 'polygons', label: 'Load Sample Zones', color: '#287094' },
    { type: 'lines', label: 'Load Sample Roads', color: '#023246' },
    { type: 'buildings', label: 'Load Sample Buildings', color: '#D4D4CE' }
  ] as const;

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border-l-4" style={{borderColor: '#287094'}}>
      <h3 className="text-xl font-bold mb-4 flex items-center" style={{color: '#023246'}}>
        <Database className="w-6 h-6 mr-2" style={{color: '#287094'}} />
        Sample Data
      </h3>
      <div className="grid grid-cols-1 gap-3">
        {sampleDataTypes.map(({ type, label, color }) => (
          <button
            key={type}
            onClick={() => loadSampleData(type)}
            className="text-white py-3 px-4 rounded-lg flex items-center space-x-2 transition-all duration-200 font-bold hover:opacity-90"
            style={{backgroundColor: color}}
          >
            <span>{label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default SampleDataSection;