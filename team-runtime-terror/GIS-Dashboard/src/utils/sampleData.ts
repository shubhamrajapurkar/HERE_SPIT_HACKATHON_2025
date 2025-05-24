import { FeatureCollection, Polygon, LineString, Point } from 'geojson';
import * as turf from '@turf/turf';

// Generate sample polygons (zones)
export const generateSamplePolygons = (): FeatureCollection => {
  const center = [72.8777, 19.0760]; // New York City coordinates
  const radius = 0.1; // 100 meters
  const numPolygons = 1;

  const features = Array.from({ length: numPolygons }, (_, i) => {
    // const angle = Math.random() * 2 * Math.PI;
    // const distance = Math.random() * radius;
    const x = center[0] + (Math.random()-0.5) * radius;
    const y = center[1] + (Math.random()-0.5) * radius;

    // Each polygon is a circle with random radius between 0.5km and 2km
    const polyRadius = 1 + Math.random() * 1.5;
    const polygon = turf.circle([x, y], polyRadius, { steps: 32, units: 'kilometers' });

    return {
      ...polygon,
      properties: {
        name: `Zone ${i + 1}`,
        area: turf.area(polygon).toFixed(2),
        type: 'residential'
      }
    };
  });

  return {
    type: 'FeatureCollection',
    features
  };
};

// Generate sample roads
export const generateSampleRoads = (): FeatureCollection => {
  const center = [72.8777, 19.0760];
  const radius = 0.05;
  const numRoads = 5;

  const features = Array.from({ length: numRoads }, (_, i) => {
    // Random start point
    const angle1 = Math.random() * 2 * Math.PI;
    const distance1 = Math.random() * radius;
    const x1 = center[0] + distance1 * Math.cos(angle1);
    const y1 = center[1] + distance1 * Math.sin(angle1);

    // Random end point
    const angle2 = Math.random() * 2 * Math.PI;
    const distance2 = Math.random() * radius;
    const x2 = center[0] + distance2 * Math.cos(angle2);
    const y2 = center[1] + distance2 * Math.sin(angle2);

    const line = turf.lineString([[x1, y1], [x2, y2]]);
    return {
      ...line,
      properties: {
        name: `Road ${i + 1}`,
        type: 'primary',
        length: turf.length(line, { units: 'kilometers' }).toFixed(2)
      }
    };
  });

  return {
    type: 'FeatureCollection',
    features
  };
};

// Generate sample buildings
export const generateSampleBuildings = (): FeatureCollection => {
  const center = [72.8777, 19.0760];
  const radius = 0.2;
  const numBuildings = 5;

  const features = Array.from({ length: numBuildings }, (_, i) => {
    // const angle = (i * 2 * Math.PI) / numBuildings;
    // const distance = Math.random() * radius;
    const x = center[0] + (Math.random()-0.5) * radius;
    const y = center[1] + (Math.random()-0.5) * radius;

    const point = turf.point([x, y]);
    return {
      ...point,
      properties: {
        name: `Building ${i + 1}`,
        type: Math.random() > 0.5 ? 'residential' : 'commercial',
        height: Math.floor(Math.random() * 50) + 10
      }
    };
  });

  return {
    type: 'FeatureCollection',
    features
  };
};