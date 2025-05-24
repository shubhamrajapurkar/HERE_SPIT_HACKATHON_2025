import React, { useState } from 'react';
import Header from './pages/Header';
import Dashboard from './pages/Dashboard';
import Pipeline from './pages/Pipeline';
import DataSources from './pages/DataSources';
import Maps from './pages/Maps';
// import QualityCheck from './pages/QualityCheck';
import GeoIntelligenceDashboard from './pages/QualityCheck';

const App = () => {
  const [activeTab, setActiveTab] = useState('dashboard');

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />;
      case 'pipeline':
        return <Pipeline />;
      case 'data':
        return <DataSources />;
      case 'maps':
        return <Maps />;
      case 'genAI Analysis':
        return <GeoIntelligenceDashboard />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="max-w-9xl mx-auto ">
        {renderContent()}
      </main>
    </div>
  );
};

export default App;