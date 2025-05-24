import { createContext, useContext, useMemo } from 'react';
import H from '@here/maps-api-for-javascript';

const HereContext = createContext(null);

export const HereProvider = ({ children }) => {
  const apikey = import.meta.env.VITE_HERE_API_KEY;
  const services = useMemo(() => {
    try {
      const platform = new H.service.Platform({
        apikey
      });
      return {
        platform,
        error: null
      };
    } catch (err) {
      return {
        platform: null,
        error: err
      };
    }
  }, [apikey]);

  if (services.error) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-gray-100">
        <div className="p-6 bg-white rounded-lg shadow-lg text-center">
          <h2 className="text-xl text-red-600 mb-2">Error</h2>
          <p>{services.error}</p>
          <p className="text-sm text-gray-600 mt-2">
            Make sure you have set VITE_API_KEY in your environment variables.
          </p>
        </div>
      </div>
    );
  }

  return (
    <HereContext.Provider value={services}>
      {children}
    </HereContext.Provider>
  );
};

export const useHere = () => {
  const context = useContext(HereContext);
  if (!context) {
    throw new Error('useHere must be used within a HereProvider');
  }
  return context;
}
