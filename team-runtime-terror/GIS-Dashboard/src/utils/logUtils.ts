// Utility function to add a log entry
export const addLogEntry = (setLogEntries, type, message) => {
  setLogEntries(prev => [
    ...prev, 
    { 
      type, 
      message, 
      timestamp: new Date().toLocaleTimeString() 
    }
  ]);
};