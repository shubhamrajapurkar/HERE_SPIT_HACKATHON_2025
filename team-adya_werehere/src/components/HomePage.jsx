import React from 'react';
import { Link } from 'react-router-dom';

const HomePage = () => {
  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-8">
            Welcome to Route Planner
          </h1>
          <div className="space-y-4">
            <Link
              to="/route"
              className="inline-block bg-blue-500 text-white px-6 py-3 rounded-lg hover:bg-blue-600 transition-colors"
            >
              Plan Your Route
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage; 