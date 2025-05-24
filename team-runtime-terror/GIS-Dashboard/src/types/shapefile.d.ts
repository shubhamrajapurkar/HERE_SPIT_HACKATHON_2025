declare module 'shapefile' {
  import { FeatureCollection } from 'geojson';

  export function read(data: ArrayBuffer): Promise<FeatureCollection>;
}