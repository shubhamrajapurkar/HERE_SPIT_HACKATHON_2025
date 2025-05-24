import * as Location from 'expo-location';
import { useEffect, useState } from 'react';

export const useUserLocation = () => {
  const [location, setLocation] = useState<Location.LocationObjectCoords | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setErrorMsg('Permission to access location was denied');
        return;
      }

      const loc = await Location.getCurrentPositionAsync({});
      setLocation(loc.coords);
    })();
  }, []);

  return { location, errorMsg };
};
