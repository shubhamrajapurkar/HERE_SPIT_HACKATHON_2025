import React, { useState, useRef } from 'react';
import { Map, Upload, Download, Trash2 } from 'lucide-react';
import * as turf from '@turf/turf';
import { Feature, FeatureCollection, Polygon, MultiPolygon, Point, LineString, GeoJsonProperties } from 'geojson';
import Canvas from './Canvas';
import LeftPanel from './LeftPanel';
import OperationsSection from './OperationsSection';
import { generateSamplePolygons, generateSampleRoads, generateSampleBuildings } from '../utils/sampleData';

type Layer = {
  id: number;
  name: string;
  type: string;
  visible: boolean;
  color: string;
  data: FeatureCollection;
};

type OperationType = 'buffer' | 'union' | 'intersection' | 'difference' | 'spatialJoin';

const GISTool: React.FC = () => {
  const [layers, setLayers] = useState<Layer[]>([]);
  const [selectedLayers, setSelectedLayers] = useState<number[]>([]);
  const [transform, setTransform] = useState({ x: 72.8777, y: 19.0760, scale: 1 });
  const [coordinates, setCoordinates] = useState({ x: 72.8777, y: 19.0760 });
  const [activityLog, setActivityLog] = useState<string[]>([]);
  const canvasRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const addToLog = (message: string, type: 'info' | 'success' | 'error' = 'info') => {
    setActivityLog(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${message}`]);
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) return;

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      try {
        const text = await file.text();
        const data = JSON.parse(text) as FeatureCollection;
        
        // Determine layer type from first feature
        const firstFeature = data.features[0];
        let type = 'unknown';
        if (firstFeature.geometry.type === 'Polygon' || firstFeature.geometry.type === 'MultiPolygon') {
          type = 'polygon';
        } else if (firstFeature.geometry.type === 'LineString' || firstFeature.geometry.type === 'MultiLineString') {
          type = 'line';
        } else if (firstFeature.geometry.type === 'Point' || firstFeature.geometry.type === 'MultiPoint') {
          type = 'point';
        }

        const newLayer: Layer = {
          id: Date.now() + i,
          name: file.name.replace(/\.[^/.]+$/, ''),
          type,
          visible: true,
          color: `#${Math.floor(Math.random()*16777215).toString(16)}`,
          data
        };

        setLayers(prev => [...prev, newLayer]);
        addToLog(`Loaded ${file.name} as ${type} layer`, 'success');
      } catch (error) {
        addToLog(`Error loading ${file.name}: ${error}`, 'error');
      }
    }
  };

  const loadSampleData = (type: 'polygon' | 'line' | 'point') => {
    let data: FeatureCollection;
    let name: string;

    switch (type) {
      case 'polygon':
        data = generateSamplePolygons();
        name = 'Sample Zones';
        break;
      case 'line':
        data = generateSampleRoads();
        name = 'Sample Roads';
        break;
      case 'point':
        data = generateSampleBuildings();
        name = 'Sample Buildings';
        break;
    }

    const newLayer: Layer = {
      id: Date.now(),
      name,
      type,
      visible: true,
      color: `#${Math.floor(Math.random()*16777215).toString(16)}`,
      data
    };

    setLayers(prev => [...prev, newLayer]);
    addToLog(`Loaded ${name}`, 'success');
  };

  const performOperation = (operation: OperationType) => {
    // For buffer, allow at least 1 selected layer; for others, require 2
    if (operation === 'buffer' && selectedLayers.length < 1) {
      addToLog('Please select at least 1 layer for buffer operation', 'error');
      return;
    }
    if (operation !== 'buffer' && selectedLayers.length < 2) {
      addToLog('Please select at least 2 layers for operation', 'error');
      return;
    }

    // Collect all features from selected layers (any geometry type)
    const selectedFeatures = selectedLayers
      .map(id => layers.find(layer => layer.id === id))
      .filter((layer): layer is Layer => layer !== undefined)
      .map(layer => layer.data.features)
      .flat();

    console.log('Selected features:', selectedFeatures);
    console.log('Geometry types:', selectedFeatures.map(f => f.geometry.type));

    // For buffer, allow at least 1 feature; for others, require 2
    if (operation === 'buffer' && selectedFeatures.length < 1) {
      addToLog('Selected layer must contain features', 'error');
      return;
    }
    if (operation !== 'buffer' && selectedFeatures.length < 2) {
      addToLog('Selected layers must contain features', 'error');
      return;
    }

    let result: Feature[] = [];

    switch (operation) {
      case 'buffer':
        // Buffer all features (works for points, lines, polygons)
        result = selectedFeatures.map(feature => {
          const buffered = turf.buffer(feature, 0.1, { units: 'kilometers' });
          // Ensure the result is a Feature<Polygon | MultiPolygon>
          if (
            buffered &&
            (buffered.geometry.type === 'Polygon' || buffered.geometry.type === 'MultiPolygon')
          ) {
            return buffered;
          } else {
            addToLog('Buffer result is not a Polygon or MultiPolygon', 'error');
            return null;
          }
        }).filter((feature): feature is Feature<Polygon | MultiPolygon> => feature !== null);
        break;

      case 'union': {
        // Only union polygons and multipolygons
        const polygonFeatures = selectedFeatures.filter(
          (feature): feature is Feature<Polygon | MultiPolygon> =>
            feature.geometry.type === 'Polygon' || feature.geometry.type === 'MultiPolygon'
        );


        if (polygonFeatures.length < 2) {
          addToLog('Union requires at least 2 polygon features', 'error');
          return;
        }


        let currentResult = polygonFeatures[0];
        for (const feature of polygonFeatures.slice(1)) {
          try {
            const unionResult = turf.union(
              currentResult as Feature<Polygon | MultiPolygon>,
              feature as Feature<Polygon | MultiPolygon>
            );
            if (unionResult) {
              currentResult = unionResult;
            } else {
              addToLog('Union operation returned null for a pair of features (possibly non-overlapping)', 'warning');
              // Continue with the previous currentResult
            }
          } catch (error) {
            addToLog(`Error performing union: ${error}`, 'error');
          }
        }
        // If union never succeeded, return all original polygons
        if (
          currentResult === polygonFeatures[0] &&
          polygonFeatures.length > 1
        ) {
          addToLog('No polygons could be unioned (possibly non-overlapping). Returning all original polygons.', 'warning');
          result = polygonFeatures;
        } else {
          result = currentResult ? [currentResult] : [];
        }
        break;
      }

      case 'intersection': {
        // Only intersect polygons and multipolygons
        const polygonFeatures = selectedFeatures.filter(
          (feature): feature is Feature<Polygon | MultiPolygon> =>
            feature.geometry.type === 'Polygon' || feature.geometry.type === 'MultiPolygon'
        );
        if (polygonFeatures.length < 2) {
          addToLog(
            `Intersection requires at least 2 polygon features. Found ${polygonFeatures.length}. 
Check that both selected layers contain valid polygons. 
If you just buffered a single feature, select another polygon layer as well.`,
            'error'
          );
          return;
        }
        let currentResult = polygonFeatures[0];
        let intersectionSucceeded = false;
        for (const feature of polygonFeatures.slice(1)) {
          try {
            const intersection_res_1 = turf.intersect(currentResult, feature);
            const union = turf.union(feature, currentResult);
            const xor = intersection_res_1 ? turf.difference(union, intersection_res_1) : union;
            const intersectionResult = turf.intersect(currentResult, intersection_res_1);
            if (intersectionResult) {
              currentResult = xor;
              intersectionSucceeded = true;
            } else {
              addToLog('Intersection operation returned null for a pair of features (possibly non-overlapping or invalid geometry)', 'warning');
              // Continue with the previous currentResult
            }
          } catch (error) {
            addToLog(`Error during intersection: ${error}`, 'error');
          }
        }
        if (intersectionSucceeded && currentResult) {
          // Return a single (multi)polygon feature if intersection succeeded
          result = [currentResult];
        } else {
          // If intersection never succeeded, return a MultiPolygon combining all geometries
          const multiPoly = {
            type: 'Feature',
            geometry: {
              type: 'MultiPolygon',
              coordinates: polygonFeatures.map(f =>
                f.geometry.type === 'Polygon'
                  ? [f.geometry.coordinates]
                  : f.geometry.coordinates
              ).flat()
            },
            properties: {}
          } as Feature<MultiPolygon>;
          addToLog('No polygons could be intersected (possibly non-overlapping or invalid). Returning MultiPolygon.', 'warning');
          result = [multiPoly];
        }
        break;
      }

      case 'difference': { 
        // Only difference polygons and multipolygons
        const polygonFeatures = selectedFeatures.filter(
          (feature): feature is Feature<Polygon | MultiPolygon> =>
            feature.geometry.type === 'Polygon' || feature.geometry.type === 'MultiPolygon'
        );
        if (polygonFeatures.length < 2) {
          addToLog(
            `Intersection requires at least 2 polygon features. Found ${polygonFeatures.length}. 
Check that both selected layers contain valid polygons. 
If you just buffered a single feature, select another polygon layer as well.`,
            'error'
          );
          return;
        }
        let currentResult = polygonFeatures[0];
        let differenceSucceeded = false;
        for (const feature of polygonFeatures.slice(1)) {
          try {
            const differenceResult = turf.difference(currentResult, feature);
            if (differenceResult) {
              currentResult = differenceResult;
              differenceSucceeded = true;
            } else {
              addToLog('Intersection operation returned null for a pair of features (possibly non-overlapping or invalid geometry)', 'warning');
              // Continue with the previous currentResult
            }
          } catch (error) {
            addToLog(`Error during intersection: ${error}`, 'error');
          }
        }
        if (differenceSucceeded && currentResult) {
          // Return a single (multi)polygon feature if intersection succeeded
          result = [currentResult];
        } else {
          // If intersection never succeeded, return a MultiPolygon combining all geometries
          const multiPoly = {
            type: 'Feature',
            geometry: {
              type: 'MultiPolygon',
              coordinates: polygonFeatures.map(f =>
                f.geometry.type === 'Polygon'
                  ? [f.geometry.coordinates]
                  : f.geometry.coordinates
              ).flat()
            },
            properties: {}
          } as Feature<MultiPolygon>;
          result = [multiPoly];
        }
        break;
      }
     
      case 'spatialJoin': {
        // Spatial join: assign attributes from polygons to points/lines that fall within them
        if (selectedLayers.length < 2) {
          addToLog('Spatial join requires at least 2 layers (e.g., points and polygons)', 'error');
          return;
        }

        // Separate features by geometry type
        const polygons = selectedFeatures.filter(
          (f): f is Feature<Polygon | MultiPolygon> =>
            f.geometry.type === 'Polygon' || f.geometry.type === 'MultiPolygon'
        );
        const pointsOrLines = selectedFeatures.filter(
          (f): f is Feature<Point | LineString> =>
            f.geometry.type === 'Point' || f.geometry.type === 'LineString'
        );

        if (polygons.length === 0 || pointsOrLines.length === 0) {
          addToLog('Spatial join requires at least one polygon and one point/line feature', 'error');
          return;
        }

        // For each point/line, find containing/intersecting polygon and merge properties
        result = pointsOrLines.map(feature => {
          const match = polygons.find(poly => {
            if (feature.geometry.type === 'Point') {
              return turf.booleanPointInPolygon(feature, poly);
            } else if (feature.geometry.type === 'LineString') {
              return turf.booleanIntersects(feature, poly);
            }
            return false;
          });
          if (match) {
            return {
              ...feature,
              properties: {
                ...feature.properties,
                ...match.properties,
                _joined: true
              }
            };
          }
          return feature;
        });

        addToLog('Spatial join completed', 'success');
        break;
      }
    }

    if (result.length > 0) {
      // Set type based on first result geometry
      let type = 'polygon';
      const geomType = result[0]?.geometry?.type;
      if (geomType === 'Point' || geomType === 'MultiPoint') type = 'point';
      else if (geomType === 'LineString' || geomType === 'MultiLineString') type = 'line';

      const newLayer: Layer = {
        id: Date.now(),
        name: `${operation} Result`,
        type,
        visible: true,
        color: `#${Math.floor(Math.random()*16777215).toString(16)}`,
        data: {
          type: 'FeatureCollection',
          features: result
        }
      };

      setLayers(prev => [...prev, newLayer]);
      addToLog(`Performed ${operation} operation successfully`, 'success');
    } else {
      addToLog(`Operation ${operation} produced no results`, 'error');
    }
  };

  const toggleLayerVisibility = (layerId: number) => {
    setLayers(prev => prev.map(layer => 
      layer.id === layerId ? { ...layer, visible: !layer.visible } : layer
    ));
  };

  const toggleLayerSelection = (layerId: number) => {
    setSelectedLayers(prev => 
      prev.includes(layerId) 
        ? prev.filter(id => id !== layerId)
        : [...prev, layerId]
    );
  };

  const removeLayer = (layerId: number) => {
    setLayers(prev => prev.filter(layer => layer.id !== layerId));
    setSelectedLayers(prev => prev.filter(id => id !== layerId));
    addToLog(`Removed layer ${layerId}`, 'info');
  };

  const clearAll = () => {
    setLayers([]);
    setSelectedLayers([]);
    setTransform({ x: 72.8777, y: 19.0760, scale: 1 });
    setActivityLog([]);
    addToLog('Cleared all layers', 'info');
  };

  const exportResults = () => {
    const resultLayers = layers.filter(layer => layer.name.includes('Result'));
    if (resultLayers.length === 0) {
      addToLog('No results to export', 'error');
      return;
    }

    resultLayers.forEach(layer => {
      const blob = new Blob([JSON.stringify(layer.data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${layer.name}.geojson`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });

    addToLog('Exported results', 'success');
  };

  const zoomIn = () => {
    setTransform(prev => ({
      ...prev,
      scale: Math.min(10, prev.scale * 1.2)
    }));
  };

  const zoomOut = () => {
    setTransform(prev => ({
      ...prev,
      scale: Math.max(0.1, prev.scale * 0.8)
    }));
  };

  const resetView = () => {
    setTransform({ x: 0, y: 0, scale: 1 });
    addToLog('Reset view', 'info');
  };

  return (
    <div className="min-h-screen font-bold" style={{background: 'linear-gradient(135deg, #F6F6F6 0%, #D4D4CE 100%)'}}>
      <div className="max-w-7xl mx-auto p-6">
        <div className="grid grid-cols-12 gap-6 h-screen">
          {/* Header */}
          <div className="col-span-12 bg-white rounded-xl shadow-lg p-4 flex items-center justify-between">
            <h1 className="text-2xl font-bold text-gray-800 flex items-center">
              <Map className="w-6 h-6 mr-2" />
              GIS Operations Tool
            </h1>
            <div className="flex space-x-2">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="bg-[#287094] text-white px-4 py-2 rounded-lg hover:bg-[#023246] transition-colors flex items-center"
              >
                <Upload className="w-4 h-4 mr-2" />
                Upload GeoJSON
              </button>
              <button
                onClick={exportResults}
                className="bg-[#287094] text-white px-4 py-2 rounded-lg hover:bg-[#023246] transition-colors flex items-center"
              >
                <Download className="w-4 h-4 mr-2" />
                Export Results
              </button>
              <button
                onClick={clearAll}
                className="bg-[#287094] text-white px-4 py-2 rounded-lg hover:bg-[#023246] transition-colors flex items-center"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear All
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".geojson"
                multiple
                onChange={handleFileUpload}
                className="hidden"
              />
            </div>
          </div>

          {/* Main Content Area */}
          <div className="col-span-12 grid grid-cols-12 gap-6">
            {/* Left Panel */}
            <div className="col-span-3 space-y-4">
              <LeftPanel
                layers={layers}
                selectedLayers={selectedLayers}
                onToggleVisibility={toggleLayerVisibility}
                onToggleSelection={toggleLayerSelection}
                onRemoveLayer={removeLayer}
                onLoadSampleData={loadSampleData}
              />
              <OperationsSection
                selectedLayers={selectedLayers}
                performOperation={performOperation}
              />
            </div>

            {/* Canvas */}
            <div className="col-span-6 h-[600px]">
              <Canvas
                canvasRef={canvasRef}
                transform={transform}
                coordinates={coordinates}
                setCoordinates={setCoordinates}
                zoomIn={zoomIn}
                zoomOut={zoomOut}
                resetView={resetView}
                layers={layers}
              />
            </div>

            {/* Activity Log */}
            <div className="col-span-3 bg-white rounded-xl shadow-lg p-4 font-mono text-sm border-l-4" style={{borderColor: '#287094'}}>
              <h3 className="text-lg font-semibold mb-2 text-[#023246]">Activity Log</h3>
              <div className="h-full overflow-y-auto">
                {activityLog.map((log, index) => (
                  <div key={index} className="text-gray-600">
                    {log}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GISTool;