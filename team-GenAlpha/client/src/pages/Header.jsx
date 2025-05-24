import React from 'react';
import { 
  BarChart3, 
  Cpu, 
  Database, 
  Map, 
  Shield,
  Bell,
  User
} from 'lucide-react';

const Header = ({ activeTab, setActiveTab }) => {
  return (
    <header className="bg-gradient-to-r from-blue-600 to-indigo-700 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Map className="h-8 w-8 text-white" />
              <h1 className="text-xl font-bold text-white">AI MapGen</h1>
            </div>
            <div className="hidden md:flex items-center space-x-1">
              <span className="px-2 py-1 bg-blue-500 text-xs text-white rounded-full">Beta</span>
            </div>
          </div>

          <nav className="hidden md:flex space-x-8">
            {[
              { key: 'dashboard', label: 'Dashboard', icon: BarChart3 },
              { key: 'pipeline', label: 'Pipeline', icon: Cpu },
              { key: 'data', label: 'Data Sources', icon: Database },
              { key: 'maps', label: 'Maps', icon: Map },
              { key: 'genAI Analysis', label: 'genAI Analysis', icon: Shield }
            ].map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={`flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === key
                    ? 'bg-blue-500 text-white'
                    : 'text-blue-100 hover:bg-blue-500 hover:text-white'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{label}</span>
              </button>
            ))}
          </nav>

          <div className="flex items-center space-x-4">
            <button className="relative p-2 text-blue-100 hover:text-white">
              <Bell className="h-5 w-5" />
              <span className="absolute top-0 right-0 h-2 w-2 bg-red-500 rounded-full"></span>
            </button>
            <button className="flex items-center space-x-2 text-blue-100 hover:text-white">
              <User className="h-5 w-5" />
              <span className="hidden md:inline">Admin</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;