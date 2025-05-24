// AppContext.js
import React, { createContext, useState } from 'react';

export const AppContext = createContext();

export const AppProvider = ({ children }) => {
  const [email, setEmail] = useState('user@example.com'); // Default email

  return (
    <AppContext.Provider value={{ email, setEmail }}>
      {children}
    </AppContext.Provider>
  );
};